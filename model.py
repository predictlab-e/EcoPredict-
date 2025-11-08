"""
model.py — алгоритмическое ядро EcoPredict (расширенный).

Содержимое:
- Модели вероятностей и обновления:
  * BayesianUpdater — байесовская апостериорная оценка
  * PlattCalibrator — калибровка вероятностей (логистическая)
  * IsotonicCalibrator — монотонная калибровка по эмпирическим данным (онлайн-накопление биннов)
  * ProbabilityCalibrator — обёртка для комбинации калибровок
  * ConfidenceInterval — доверительные интервалы (радиус/апостериорные Beta-граничные оценки)
- Временные компоненты:
  * TemporalDecay — экспоненциальный декей по возрасту сигнала
  * HorizonManager — связь горизонтов с half-life, нормализация
- Риск и штрафы:
  * VolatilityPenalty — пенальти за волатильность
  * RiskManager — сводный риск-фильтр (маржа безопасности, вола, ликвидность, drawdown proxy)
- Ансамбль:
  * EnsembleCombiner — взвешивание вероятностей/оценок мягким softmax
- Дополнительные утилиты:
  * RobustStats — устойчивые метрики (медиана, MAD)
  * OnlineBinning — полезный класс для накопления эмпирических кривых калибровки
  * Smoothers — сглаживатели (EMA, Holt-like)
  * TrendEstimators — простые модели тренда (линейная регрессия, STL-подобная декомпозиция)
  * SeasonalityEstimators — простая сезонность для интроспекции

Зависимости:
- utils.py: bounded, exp_weight, stable_softmax, safe_div, normalize_01, robust_median, rolling_mean, rolling_std
"""

from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from utils import (
    bounded,
    exp_weight,
    stable_softmax,
    safe_div,
    normalize_01,
    robust_median,
    rolling_mean,
    rolling_std,
)

# --------------------------------------------------------------------------------------
# Модели вероятностей и калибровка
# --------------------------------------------------------------------------------------

class BayesianUpdater:
    """
    Байесовское обновление вероятности на основе Beta(alpha, beta).
    - update(p, weight): обновляет параметры с учётом нового наблюдения (псевдо-наблюдение).
    - alpha, beta интерпретируются как априорные счётчики успехов/неудач.

    Подход:
    - p ∈ [0,1] — сигнал вероятности (например, линейная смесь признаков).
    - weight ≥ 0 — «масса» наблюдения (чем выше, тем сильнее апдейт).
    - Возвращаем апостериорную оценку p_post = alpha/(alpha+beta) и состояние.
    """

    def __init__(self, alpha: float = 5.0, beta: float = 5.0, clip: Tuple[float, float] = (0.01, 0.99)):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.clip = clip

    def update(self, p_signal: float, weight: float = 1.0) -> Tuple[float, Dict[str, float]]:
        p = bounded(p_signal, *self.clip)
        w = max(0.0, float(weight))
        # Псевдо-наблюдение:
        self.alpha += p * w
        self.beta += (1.0 - p) * w
        posterior = bounded(self.alpha / (self.alpha + self.beta), *self.clip)
        state = {"alpha": self.alpha, "beta": self.beta, "posterior": posterior}
        return posterior, state


class PlattCalibrator:
    """
    Параметрическая калибровка (Platt scaling): p' = sigmoid(a * logit(p) + b)
    Здесь используем упрощённый вариант: p' = bounded(slope * p + bias).
    Примечание: для производственной настройки a,b требуется обучать на отложенных данных.

    Мы оставляем «логистическую» форму простой линейной калибровки из-за отсутствия обученного набора.
    """

    def __init__(self, slope: float = 0.95, bias: float = 0.02, clip: Tuple[float, float] = (0.01, 0.99)):
        self.slope = float(slope)
        self.bias = float(bias)
        self.clip = clip

    def calibrate(self, p: float) -> float:
        return bounded(self.slope * p + self.bias, *self.clip)


class OnlineBinning:
    """
    Онлайн-накопитель биннов (для IsotonicCalibrator).
    Держит массив из N биннов, каждый с суммой вероятностей и количеством наблюдений.
    Предполагаем, что входные p в [0,1]. Сопоставляем p → bin_idx.

    Можно использовать для эмпирической калибровки: для каждого бинна вычисляем пропорцию «успехов»,
    но здесь мы работаем без истинной метки (ground truth), поэтому моделируем «успех» как p.
    """

    def __init__(self, bins: int = 20, min_bin_size: int = 30):
        self.bins = max(5, int(bins))
        self.min_bin_size = max(1, int(min_bin_size))
        self.counts = [0] * self.bins
        self.sums = [0.0] * self.bins

    def add(self, p: float):
        i = self._index(p)
        self.counts[i] += 1
        self.sums[i] += p

    def estimate(self, p: float) -> float:
        i = self._index(p)
        if self.counts[i] < self.min_bin_size:
            # fallback к соседям
            left = max(0, i - 1)
            right = min(self.bins - 1, i + 1)
            vals = []
            for idx in (left, i, right):
                if self.counts[idx] > 0:
                    vals.append(self.sums[idx] / self.counts[idx])
            if not vals:
                return p
            return sum(vals) / len(vals)
        else:
            return self.sums[i] / self.counts[i]

    def _index(self, p: float) -> int:
        p = bounded(p, 0.0, 1.0)
        idx = int(p * self.bins)
        return min(self.bins - 1, max(0, idx))


class IsotonicCalibrator:
    """
    Монотонная калибровка: мы используем OnlineBinning → оцениваем p' как среднее по бинну и сглаживаем.
    Реальный изотонический регрессор строит кусочно-постоянную монотонную функцию,
    но в проде часто используют эмпирику с биннингом + сглаживание.

    Здесь:
    - add_observation(p): накопление p (условно)
    - calibrate(p): возвращает скорректированное значение из биннов
    """

    def __init__(self, bins: int = 20, min_bin_size: int = 30, clip: Tuple[float, float] = (0.01, 0.99)):
        self.binner = OnlineBinning(bins=bins, min_bin_size=min_bin_size)
        self.clip = clip
        # Инициализация равномерными наблюдениями, чтобы не было пустых биннов на старте
        for _ in range(5 * bins):
            self.binner.add(random.random())

    def add_observation(self, p: float):
        self.binner.add(bounded(p, *self.clip))

    def calibrate(self, p: float) -> float:
        p = bounded(p, *self.clip)
        return bounded(self.binner.estimate(p), *self.clip)


class ProbabilityCalibrator:
    """
    Комбинированная калибровка: сначала Platt (гладкая линейная поправка), затем Isotonic (монотонная).
    """

    def __init__(self, platt: PlattCalibrator, isotonic: IsotonicCalibrator):
        self.platt = platt
        self.isotonic = isotonic

    def calibrate(self, p: float) -> float:
        return self.isotonic.calibrate(self.platt.calibrate(p))


class ConfidenceInterval:
    """
    Доверительный интервал для вероятности:
    - Простой вариант: фиксированный радиус.
    - Расширенный вариант (опционально): апостериорные границы Beta-распределения.

    Здесь реализуем базово: radius и клип в [0,1].
    """

    def __init__(self, radius: float = 0.08):
        self.radius = float(radius)

    def bounds(self, p: float) -> Tuple[float, float]:
        lo = bounded(p - self.radius, 0.0, 1.0)
        hi = bounded(p + self.radius, 0.0, 1.0)
        return lo, hi

# --------------------------------------------------------------------------------------
# Временные компоненты
# --------------------------------------------------------------------------------------

class TemporalDecay:
    """
    Экспоненциальный декей по возрасту сигнала:
    p' = w * p + (1 - w) * 0.5, где w = 0.5^(age/half_life)

    Возвращает:
    - скорректированную вероятность
    - вес (recency) ∈ [0,1], который можно логировать как степень «новизны».
    """

    def apply(self, p: float, age_minutes: float, half_life: float = 120.0) -> Tuple[float, float]:
        w = exp_weight(age_minutes, half_life)
        p_adj = bounded(w * p + (1.0 - w) * 0.5, 0.01, 0.99)
        return p_adj, w


class HorizonManager:
    """
    Управляет параметрами, связанными с горизонтом:
    - half_life(horizon): перевод горизонта в понятный half-life для TemporalDecay.
    - normalize(horizon): нормализация веса горизонта (например, короткие горизонты чуть агрессивнее).
    """

    def __init__(self, h2m: Dict[str, int]):
        self.h2m = dict(h2m)

    def half_life(self, horizon: str) -> float:
        minutes = float(self.h2m.get(horizon, 120))
        # Простая эвристика: half-life как половина горизонта
        return max(30.0, minutes / 2.0)

    def normalize(self, horizon: str) -> float:
        minutes = float(self.h2m.get(horizon, 120))
        # Чем меньше minutes, тем выше нормирующий коэффициент
        return bounded(240.0 / max(60.0, minutes), 0.5, 1.5)

# --------------------------------------------------------------------------------------
# Риск и штрафы
# --------------------------------------------------------------------------------------

class VolatilityPenalty:
    """
    Пенальти за волатильность: p' = p - scale * f(volatility)
    где f — нормированная функция волатильности, например f(v) = v / (v + v0).

    Идея: при высокой волатильности мы уменьшаем уверенность в покупке.
    """

    def __init__(self, scale: float = 0.15, v0: float = 0.05):
        self.scale = float(scale)
        self.v0 = float(v0)

    def apply(self, p: float, vol: float) -> float:
        v_norm = vol / (vol + self.v0)
        return bounded(p - self.scale * v_norm, 0.01, 0.99)


class RiskManager:
    """
    Сводный риск-фильтр:
    - Безопасная маржа: p' = p - margin
    - Пенальти за волу: уже применены VolatilityPenalty (но можно добавить ещё)
    - Прокси drawdown: если нижняя граница CI низкая, уменьшаем
    - Минимальная уверенность: если ci_high - ci_low < min_confidence, уменьшаем
    - Ликвидность (если дана): низкая ликвидность — доп. пенальти

    config:
    {
        "prob_safety_margin": 0.05,
        "max_leverage": 1.0,
        "drawdown_penalty": 0.07,
        "vol_penalty_scale": 0.15,
        "min_confidence": 0.05,
    }
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = dict(config)

    def apply(self, prob: float, summary_by_h: Dict[str, Dict[str, Any]]) -> float:
        p = prob
        # 1) Маржа безопасности
        p -= self.cfg.get("prob_safety_margin", 0.05)

        # 2) Минимальная уверенность и drawdown proxy: смотрим по горизонтам
        min_ci_width = 1.0
        min_ci_low = 1.0
        for h, out in summary_by_h.items():
            ci_low = float(out["summary"]["ci_low"])
            ci_high = float(out["summary"]["ci_high"])
            width = ci_high - ci_low
            min_ci_width = min(min_ci_width, width)
            min_ci_low = min(min_ci_low, ci_low)

        if min_ci_width < self.cfg.get("min_confidence", 0.05):
            p -= 0.03  # понижаем чуть-чуть за недостаточную уверенность

        # drawdown proxy: если нижняя граница где-то очень низкая
        if min_ci_low < 0.2:
            p -= self.cfg.get("drawdown_penalty", 0.07)

        # Климат по ликвидности: если есть признак ликвидности в факторах — применим тонкую коррекцию
        # В summary_by_h у нас нет факторов, поэтому этот блок оставим как идейный.
        # В случае расширения, сюда можно передавать факторные веса и агрегировать.

        return bounded(p, 0.01, 0.99)

# --------------------------------------------------------------------------------------
# Ансамбль
# --------------------------------------------------------------------------------------

class EnsembleCombiner:
    """
    Комбинирует набор вероятностей/оценок с помощью «стабильного» softmax.
    - temp: температура — чем выше, тем более равномерное распределение весов.

    Методы:
    - combine(probs): возвращает взвешенную сумму
    - weights(probs): возвращает веса
    """

    def __init__(self, temp: float = 1.0):
        self.temp = float(temp)

    def combine(self, probs: List[float]) -> float:
        w = stable_softmax(probs, temp=self.temp)
        return bounded(sum(p * wi for p, wi in zip(probs, w)), 0.01, 0.99)

    def weights(self, probs: List[float]) -> List[float]:
        return list(stable_softmax(probs, temp=self.temp))

# --------------------------------------------------------------------------------------
# Дополнительные утилиты: устойчивые метрики и сглаживание
# --------------------------------------------------------------------------------------

class RobustStats:
    """
    Устойчивые статистики для анализа:
    - median: робастная медиана
    - mad: median absolute deviation
    - iqr: interquartile range (оценка по квартилям)
    """

    @staticmethod
    def median(x: List[float]) -> float:
        if not x:
            return 0.0
        return robust_median(x)

    @staticmethod
    def mad(x: List[float], scale: float = 1.4826) -> float:
        if not x:
            return 0.0
        m = robust_median(x)
        devs = [abs(xi - m) for xi in x]
        return scale * robust_median(devs)

    @staticmethod
    def iqr(x: List[float]) -> float:
        if not x:
            return 0.0
        xs = sorted(x)
        n = len(xs)
        q1 = xs[int(0.25 * (n - 1))]
        q3 = xs[int(0.75 * (n - 1))]
        return q3 - q1


class Smoothers:
    """
    Сглаживающие функции:
    - ema(x, alpha): экспоненциальное сглаживание
    - double_ema(x, alpha_fast, alpha_slow): комбинированное сглаживание
    - holt_like(x, alpha, beta): упрощённый Holt-подобный трендовый сглаживатель
    """

    @staticmethod
    def ema(x: List[float], alpha: float = 0.2) -> List[float]:
        if not x:
            return []
        out = []
        last = x[0]
        a = bounded(alpha, 0.0, 1.0)
        for xi in x:
            last = a * xi + (1.0 - a) * last
            out.append(last)
        return out

    @staticmethod
    def double_ema(x: List[float], alpha_fast: float = 0.35, alpha_slow: float = 0.1) -> List[float]:
        fast = Smoothers.ema(x, alpha_fast)
        slow = Smoothers.ema(x, alpha_slow)
        return [bounded(0.6 * f + 0.4 * s, 0.0, 1.0) for f, s in zip(fast, slow)]

    @staticmethod
    def holt_like(x: List[float], alpha: float = 0.3, beta: float = 0.1) -> List[float]:
        """
        Упрощённый Holt: уровень + тренд.
        """
        if not x:
            return []
        level = x[0]
        trend = 0.0
        a = bounded(alpha, 0.0, 1.0)
        b = bounded(beta, 0.0, 1.0)
        out = []
        for xi in x:
            prev_level = level
            level = a * xi + (1.0 - a) * (level + trend)
            trend = b * (level - prev_level) + (1.0 - b) * trend
            out.append(level)
        return out

# --------------------------------------------------------------------------------------
# Тренд/сезонность — простые вспомогательные модели (для расширений)
# --------------------------------------------------------------------------------------

class TrendEstimators:
    """
    Простые оценки тренда:
    - linear_beta(x): коэффициент наклона линейной регрессии по времени
    - momentum_score(x): нормированный моментум последнего значения относительно среднего
    """

    @staticmethod
    def linear_beta(x: List[float]) -> float:
        if not x:
            return 0.0
        n = len(x)
        t_idx = list(range(n))
        mean_t = rolling_mean(t_idx)
        mean_x = rolling_mean(x)
        cov_tx = rolling_mean([(ti - mean_t) * (xi - mean_x) for ti, xi in zip(t_idx, x)])
        var_t = rolling_mean([(ti - mean_t) ** 2 for ti in t_idx]) + 1e-6
        return bounded(cov_tx / var_t, -1.0, 1.0)

    @staticmethod
    def momentum_score(x: List[float]) -> float:
        if not x:
            return 0.0
        avg = rolling_mean(x)
        std = rolling_std(x) + 1e-6
        return bounded((x[-1] - avg) / std, -5.0, 5.0)


class SeasonalityEstimators:
    """
    Сезонность (очень простая, декоративная):
    - sinusoid(len, period): синусообразная компонента по длине ряда
    """

    @staticmethod
    def sinusoid(n: int, period: float = 12.0, amp: float = 0.1) -> float:
        if n <= 0:
            return 0.0
        return bounded(math.sin(n / period) * amp, -0.5, 0.5)

# --------------------------------------------------------------------------------------
# Высокоуровневые утилиты модели
# --------------------------------------------------------------------------------------

def linear_feature_mix(features: Dict[str, float], weights: Dict[str, float], clip: Tuple[float, float] = (0.01, 0.99)) -> float:
    """
    Линейная смесь признаков и базовой вероятности.
    """
    base = 0.5
    for k, w in weights.items():
        base += w * float(features.get(_alias(k), 0.0))
    return bounded(base, *clip)

def _alias(feature_name: str) -> str:
    """
    Сопоставление ключей весов с ключами признаков (на случай отличий).
    """
    m = {
        "momentum": "zscore_momentum",
        "roc": "roc",
        "volatility": "volatility",
        "volume_spike": "volume_spike",
        "order_imbalance": "order_imbalance",
        "trend_strength": "trend_strength",
        "mean_reversion": "mean_reversion",
        "seasonality": "seasonality",
        "liquidity": "liquidity",
    }
    return m.get(feature_name, feature_name)

# --------------------------------------------------------------------------------------
# Демонстрационная стохастическая подмодель (для ансамблей)
# --------------------------------------------------------------------------------------

class ProbabilisticSubModel:
    """
    Дополнительная подмодель, генерирующая скор по признакам:
    - stochastic tweak: добавляем шум к линейной смеси для разнообразия ансамбля.
    """

    def __init__(self, weights: Dict[str, float], noise_scale: float = 0.02):
        self.weights = dict(weights)
        self.noise_scale = float(noise_scale)

    def predict(self, features: Dict[str, float]) -> float:
        p = linear_feature_mix(features, self.weights)
        noise = (random.random() - 0.5) * 2.0 * self.noise_scale
        return bounded(p + noise, 0.01, 0.99)

# --------------------------------------------------------------------------------------
# Высокоуровневый фасад для модели (используется в Single.py)
# --------------------------------------------------------------------------------------

class ModelFacade:
    """
    Композитная модель, объединяющая:
    - линейную смесь признаков
    - байесовское обновление
    - временной декей
    - калибровку (Platt + Isotonic)
    - волатильностный пенальти
    - доверительный интервал
    - ансамбль подмоделей (опционально)

    Этот фасад не используется напрямую в Single.py, но может пригодиться для расширений.
    """

    def __init__(
        self,
        base_weights: Dict[str, float],
        bayes: Optional[BayesianUpdater] = None,
        tdecay: Optional[TemporalDecay] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
        vol_penalty: Optional[VolatilityPenalty] = None,
        ci: Optional[ConfidenceInterval] = None,
        ensemble_temp: float = 1.0,
        use_submodels: bool = True,
    ):
        self.base_weights = dict(base_weights)
        self.bayes = bayes or BayesianUpdater()
        self.tdecay = tdecay or TemporalDecay()
        self.calibrator = calibrator or ProbabilityCalibrator(PlattCalibrator(), IsotonicCalibrator())
        self.vol_penalty = vol_penalty or VolatilityPenalty()
        self.ci = ci or ConfidenceInterval()
        self.ensemble = EnsembleCombiner(temp=ensemble_temp)
        self.use_submodels = use_submodels

        # Инициализируем набор субмоделей для ансамбля (вариативность весов)
        self.submodels: List[ProbabilisticSubModel] = []
        if use_submodels:
            w1 = {k: v * 1.05 for k, v in self.base_weights.items()}
            w2 = {k: v * 0.95 for k, v in self.base_weights.items()}
            w3 = {k: (v * (1.0 if "momentum" in k else 0.9)) for k, v in self.base_weights.items()}
            self.submodels = [
                ProbabilisticSubModel(w1, noise_scale=0.015),
                ProbabilisticSubModel(w2, noise_scale=0.025),
                ProbabilisticSubModel(w3, noise_scale=0.020),
            ]

    def evaluate(self, features: Dict[str, float], age_minutes: float, half_life: float) -> Dict[str, Any]:
        # База
        base_prob = linear_feature_mix(features, self.base_weights)
        p_bayes, bayes_state = self.bayes.update(base_prob, weight=3.0)

        # Время
        p_time, recency = self.tdecay.apply(p_bayes, age_minutes=age_minutes, half_life=half_life)

        # Ансамбль субмоделей
        probs = [p_time]
        for sm in self.submodels:
            probs.append(sm.predict(features))
        p_ens = self.ensemble.combine(probs)

        # Калибровка и пенальти
        p_cal = self.calibrator.calibrate(p_ens)
        p_vol = self.vol_penalty.apply(p_cal, vol=float(features.get("volatility", 0.0)))

        # ДИ
        lo, hi = self.ci.bounds(p_vol)

        summary = {
            "prob": p_vol,
            "ci_low": lo,
            "ci_high": hi,
            "recency": recency,
            "bayes": bayes_state,
        }
        factors = {
            "momentum": float(features.get("zscore_momentum", 0.0)),
            "roc": float(features.get("roc", 0.0)),
            "volatility": float(features.get("volatility", 0.0)),
            "volume": float(features.get("volume_spike", 0.0)),
            "order_imbalance": float(features.get("order_imbalance", 0.0)),
            "trend_strength": float(features.get("trend_strength", 0.0)),
            "mean_reversion": float(features.get("mean_reversion", 0.0)),
            "seasonality": float(features.get("seasonality", 0.0)),
            "liquidity": float(features.get("liquidity", 0.5)),
            "recency": recency,
        }
        return {"summary": summary, "factor_weights": factors}

# --------------------------------------------------------------------------------------
# Интерфейс, который использует Single.py
# --------------------------------------------------------------------------------------

# Для совместимости с Single.py
# Single.py импортирует: BayesianUpdater, ProbabilityCalibrator, TemporalDecay, VolatilityPenalty,
# HorizonManager, EnsembleCombiner, RiskManager, ConfidenceInterval, PlattCalibrator, IsotonicCalibrator.

# Все классы уже объявлены выше и готовы к использованию.

# --------------------------------------------------------------------------------------
# Дополнение: расширенные доверительные интервалы (опция)
# --------------------------------------------------------------------------------------

class BetaConfidenceInterval(ConfidenceInterval):
    """
    Альтернативный вариант построения CI для вероятности, опирающийся на апостериорные параметры Beta.
    Если известны alpha, beta (из BayesianUpdater), можно оценить интервалы Клаббеза–Пирсона или аппроксимации.
    Здесь оставим упрощённую аппроксимацию по нормальному приближению:
      var ≈ (alpha * beta) / [(alpha + beta)^2 * (alpha + beta + 1)]
      sd = sqrt(var)
      ci = p ± z * sd, z ≈ 1.96 для 95%

    Примечание: для строгих оценок лучше использовать специализированные функции/библиотеки.
    """

    def __init__(self, z: float = 1.96):
        super().__init__(radius=0.0)  # отключаем фиксированный радиус
        self.z = float(z)

    def bounds_with_beta(self, p: float, alpha: float, beta: float) -> Tuple[float, float]:
        total = alpha + beta
        if total <= 1.0:
            # fallback к базовому радиусу
            return super().bounds(p)
        var = (alpha * beta) / (total * total * (total + 1.0))
        sd = math.sqrt(max(1e-9, var))
        lo = bounded(p - self.z * sd, 0.0, 1.0)
        hi = bounded(p + self.z * sd, 0.0, 1.0)
        return lo, hi

# --------------------------------------------------------------------------------------
# Тестовые заглушки/демо (можно удалить в проде)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Демонстрация работы отдельных компонентов
    feats_demo = {
        "zscore_momentum": 0.8,
        "volatility": 0.12,
        "roc": 0.05,
        "volume_spike": 0.3,
        "order_imbalance": 0.1,
        "trend_strength": 0.2,
        "mean_reversion": -0.4,
        "seasonality": 0.05,
        "liquidity": 0.7,
        "age_minutes": 30.0,
    }

    base_weights_demo = {
        "momentum": 0.23,
        "roc": 0.17,
        "volatility": -0.12,
        "volume_spike": 0.11,
        "order_imbalance": 0.08,
        "trend_strength": 0.12,
        "mean_reversion": -0.05,
        "seasonality": 0.06,
        "liquidity": 0.07,
    }

    bayes = BayesianUpdater(alpha=5, beta=5)
    tdecay = TemporalDecay()
    platt = PlattCalibrator(slope=0.95, bias=0.02)
    iso = IsotonicCalibrator(bins=20, min_bin_size=30)
    calibrator = ProbabilityCalibrator(platt, iso)
    vol_pen = VolatilityPenalty(scale=0.15)
    ci = ConfidenceInterval(radius=0.08)
    hmgr = HorizonManager({"1h": 60, "3h": 180, "6h": 360, "24h": 1440})

    # Фасадная модель
    facade = ModelFacade(
        base_weights=base_weights_demo,
        bayes=bayes,
        tdecay=tdecay,
        calibrator=calibrator,
        vol_penalty=vol_pen,
        ci=ci,
        ensemble_temp=1.0,
        use_submodels=True,
    )

    out = facade.evaluate(feats_demo, age_minutes=feats_demo["age_minutes"], half_life=hmgr.half_life("3h"))
    print("Facade output:", out)

    # Проверка EnsembleCombiner
    ens = EnsembleCombiner(temp=1.0)
    probs = [0.55, 0.62, 0.48, 0.51]
    print("Ensemble combine:", ens.combine(probs), "weights:", ens.weights(probs))

    # Проверка BetaConfidenceInterval
    beta_ci = BetaConfidenceInterval(z=1.96)
    p = 0.6
    alpha, beta_val = 25, 20
    print("Beta CI:", beta_ci.bounds_with_beta(p, alpha, beta_val))
