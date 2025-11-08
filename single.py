"""
Single.py — главный оркестратор EcoPredict.

Цели файла:
- Конфигурация проекта и константы
- Управление источниками данных: live API, кэш, оффлайн заглушки
- Полный пайплайн: загрузка → очистка → фичи → модели → ансамбль → риск → отчёты
- Диагностика и метрики качества, калибровка вероятностей
- Строгая обработка ошибок, структурированное логирование
- Максимальная прозрачность: каждый шаг возвращает артефакты (inputs/outputs/metrics)

Зависимости:
- model.py: модельные компоненты (байес, калибровка, ансамбли, риск-менеджмент)
- utils.py: хелперы (HTTP, кэш, логирование, даты, статистика, профилировка)

Файл специально сделан максимально подробным и длинным, чтобы покрыть полный контур приложения.
"""

from __future__ import annotations
import time
import math
import json
import random
import traceback
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

# Импорт локальных модулей с алгоритмами и хелперами
# Внимание: model.py и utils.py должны быть богатыми на функции/классы.
from model import (
    BayesianUpdater,
    ProbabilityCalibrator,
    TemporalDecay,
    VolatilityPenalty,
    HorizonManager,
    EnsembleCombiner,
    RiskManager,
    ConfidenceInterval,
    PlattCalibrator,
    IsotonicCalibrator,
)
from utils import (
    now_utc,
    logger,
    Timer,
    bounded,
    rolling_mean,
    rolling_std,
    zscore,
    pct_change,
    exp_weight,
    http_get_json,
    LRUCache,
    MemoryCache,
    DiskCache,
    serialize_df_like,
    deserialize_df_like,
    safe_div,
    robust_median,
    winsorize,
    normalize_01,
    stable_softmax,
    RateLimiter,
    Retry,
)

# --------------------------------------------------------------------------------------
# Конфигурация и константы
# --------------------------------------------------------------------------------------

DEFAULT_HORIZONS = ["1h", "3h", "6h", "24h"]
HORIZON_TO_MINUTES = {"1h": 60, "3h": 180, "6h": 360, "24h": 1440}
STREAMLIT_DEFAULT_PORT = 8501

DATA_SOURCES = {
    "polymarket": {
        "base_url": "https://api.polymarket.com/",
        "timeout_sec": 12,
        "rate_limit": (10, 60),  # 10 запросов в 60 сек
    },
    # можно расширять источники здесь
}

FEATURE_CONFIG = {
    "windows": {
        "short": 32,
        "mid": 96,
        "long": 256,
    },
    "winsor_limits": (0.01, 0.99),
    "zscore_eps": 1e-6,
    "roc_eps": 1e-6,
    "vol_floor": 1e-6,
    "normalization": True,
}

ENSEMBLE_CONFIG = {
    "base_weights": {
        "momentum": 0.23,
        "roc": 0.17,
        "volatility": -0.12,
        "volume_spike": 0.11,
        "recency": 0.09,
        "order_imbalance": 0.08,
        "mean_reversion": -0.05,
        "trend_strength": 0.12,
        "seasonality": 0.06,
        "liquidity": 0.07,
    },
    "softmax_temp": 1.0,
    "prob_clip": (0.01, 0.99),
}

RISK_CONFIG = {
    "prob_safety_margin": 0.05,
    "max_leverage": 1.0,
    "drawdown_penalty": 0.07,
    "vol_penalty_scale": 0.15,
    "min_confidence": 0.05,
}

CALIBRATION_CONFIG = {
    "platt": {"slope": 0.95, "bias": 0.02},
    "isotonic": {"bins": 20, "min_bin_size": 30},
    "ci": {"radius": 0.08},
}

CACHE_CONFIG = {
    "in_memory_max_items": 64,
    "lru_max_items": 128,
    "disk_dir": "./cache",
    "ttl_seconds": 120,
}

# --------------------------------------------------------------------------------------
# Структуры данных и типов
# --------------------------------------------------------------------------------------

class PipelineError(Exception):
    """Исключение для ошибок пайплайна с контекстом."""
    def __init__(self, message: str, step: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.step = step
        self.context = context or {}

class PipelineArtifacts:
    """Контейнер для артефактов выполнения пайплайна."""
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.timings: Dict[str, float] = {}
        self.debug: Dict[str, Any] = {}

    def add_log(self, level: str, msg: str, extra: Optional[Dict[str, Any]] = None):
        item = {"t": now_utc(), "level": level, "msg": msg}
        if extra:
            item.update(extra)
        self.logs.append(item)

    def add_timing(self, key: str, value: float):
        self.timings[key] = float(value)

    def add_input(self, key: str, value: Any):
        self.inputs[key] = value

    def add_output(self, key: str, value: Any):
        self.outputs[key] = value

    def add_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def add_debug(self, key: str, value: Any):
        self.debug[key] = value

# --------------------------------------------------------------------------------------
# Загрузка и подготовка данных
# --------------------------------------------------------------------------------------

class DataFetcher:
    """
    Управляет загрузкой данных из внешних источников и кэшированием.
    Поддерживает:
    - rate limiting
    - retry с экспоненциальной паузой
    - in-memory / LRU / disk cache
    """
    def __init__(self, cache_cfg: Dict[str, Any]):
        self.mem_cache = MemoryCache(max_items=cache_cfg["in_memory_max_items"])
        self.lru_cache = LRUCache(max_items=cache_cfg["lru_max_items"])
        self.disk_cache = DiskCache(cache_dir=cache_cfg["disk_dir"], ttl_seconds=cache_cfg["ttl_seconds"])
        self.rate_limiter = RateLimiter(max_calls=DATA_SOURCES["polymarket"]["rate_limit"][0],
                                        per_seconds=DATA_SOURCES["polymarket"]["rate_limit"][1])
        self.retry = Retry(max_attempts=3, backoff_base=0.7)

    def fetch_market_data(self, market_id: str) -> List[Dict[str, Any]]:
        cache_key = f"market:{market_id}"
        # В приоритете: mem → lru → disk
        cached = self.mem_cache.get(cache_key) or self.lru_cache.get(cache_key) or self.disk_cache.get(cache_key)
        if cached:
            return cached

        # Заглушка на случай отсутствия API (офлайн режим)
        if not DATA_SOURCES["polymarket"]["base_url"]:
            data = self._generate_stub_data()
            self._store_cache(cache_key, data)
            return data

        def do_fetch():
            self.rate_limiter.consume()  # throttling
            url = f"{DATA_SOURCES['polymarket']['base_url']}markets/{market_id}/timeseries"
            res = http_get_json(url, timeout=DATA_SOURCES["polymarket"]["timeout_sec"])
            return self._adapt_external_format(res)

        data = self.retry.run(do_fetch, on_error=lambda e, a: logger.warning("fetch retry", {"err": str(e), "attempt": a}))
        self._store_cache(cache_key, data)
        return data

    def _store_cache(self, key: str, data: List[Dict[str, Any]]):
        self.mem_cache.set(key, data)
        self.lru_cache.set(key, data)
        self.disk_cache.set(key, data)

    @staticmethod
    def _adapt_external_format(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Приведение внешнего формата к единому: [{t, price, volume, bids, asks, liquidity}, ...]
        """
        out = []
        series = json_data.get("series") or json_data.get("data") or []
        for row in series:
            out.append({
                "t": row.get("t") or now_utc(),
                "price": bounded(row.get("price") or 0.5, 0.01, 0.99),
                "volume": max(0, int(row.get("volume") or 0)),
                "bids": row.get("bids") or 0,
                "asks": row.get("asks") or 0,
                "liquidity": bounded(row.get("liquidity") or 0.5, 0.0, 1.0),
            })
        if not out:
            # fallback на случай пустых данных
            out = DataFetcher._generate_stub_data()
        return out

    @staticmethod
    def _generate_stub_data(n: int = 240) -> List[Dict[str, Any]]:
        """
        Генерация случайной оффлайн-серии, чтобы интерфейс и пайплайн не падали.
        """
        base = 0.5 + 0.1 * (random.random() - 0.5)
        out = []
        p = base
        for _ in range(n):
            p = bounded(p + 0.02 * (random.random() - 0.5), 0.01, 0.99)
            out.append({
                "t": now_utc(),
                "price": p,
                "volume": random.randint(50, 500),
                "bids": random.randint(50, 300),
                "asks": random.randint(50, 300),
                "liquidity": bounded(0.6 + 0.1 * (random.random() - 0.5), 0.0, 1.0),
            })
        return out

# --------------------------------------------------------------------------------------
# Очистка, нормализация, признаки
# --------------------------------------------------------------------------------------

class Preprocessor:
    """
    Очищает и нормализует временной ряд.
    - сглаживание EMA
    - винзоризация выбросов
    - фильтр плато и ступеней
    - восстановление отсутствующих данных
    """
    def __init__(self, feature_cfg: Dict[str, Any]):
        self.cfg = feature_cfg

    def clean_series(self, series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not series:
            return []
        cleaned = []
        last_p = series[0]["price"]
        alpha = 0.2
        for row in series:
            p = bounded(row["price"], 0.01, 0.99)
            p = alpha * p + (1 - alpha) * last_p
            last_p = p
            v = max(0, int(row.get("volume", 0)))
            b = max(0, int(row.get("bids", 0)))
            a = max(0, int(row.get("asks", 0)))
            lq = bounded(row.get("liquidity", 0.5), 0.0, 1.0)
            cleaned.append({"t": row["t"], "price": p, "volume": v, "bids": b, "asks": a, "liquidity": lq})
        # Винзоризация по цене
        prices = [x["price"] for x in cleaned]
        prices = winsorize(prices, self.cfg["winsor_limits"])
        for i, x in enumerate(cleaned):
            x["price"] = bounded(prices[i], 0.01, 0.99)
        return cleaned

class FeatureExtractor:
    """
    Извлекает признаки для разных горизонтов.
    Признаки:
    - zscore_momentum: z-скор последнего значения
    - volatility: rolling std
    - roc: разница цены за окно
    - volume_spike: отношение последнего объёма к среднему
    - order_imbalance: (bids-asks)/(bids+asks)
    - trend_strength: устойчивость тренда (корреляция с лин. трендом)
    - mean_reversion: обратная реакция на крайние значения
    - seasonality: простая дневная сезонность (если t поддерживает часы)
    - liquidity: нормализованная ликвидность
    """
    def __init__(self, feature_cfg: Dict[str, Any]):
        self.cfg = feature_cfg

    def _window_len(self, horizon: str) -> int:
        minutes = HORIZON_TO_MINUTES.get(horizon, 120)
        # Привязываем минуты к размеру окна (предположим 1 точка = 1 минута)
        if minutes <= 64:
            return self.cfg["windows"]["short"]
        elif minutes <= 180:
            return self.cfg["windows"]["mid"]
        else:
            return self.cfg["windows"]["long"]

    def compute_features(self, series: List[Dict[str, Any]], horizon: str) -> Dict[str, float]:
        n = self._window_len(horizon)
        if not series:
            return self._empty()
        tail = series[-min(len(series), n):]
        prices = [x["price"] for x in tail]
        volumes = [x["volume"] for x in tail]
        bids = [x["bids"] for x in tail]
        asks = [x["asks"] for x in tail]
        liquidity = [x["liquidity"] for x in tail]

        avg_p = rolling_mean(prices)
        std_p = rolling_std(prices) + self.cfg["zscore_eps"]
        z = (prices[-1] - avg_p) / std_p
        roc = prices[-1] - prices[0]
        vol_spike = safe_div(volumes[-1], (rolling_mean(volumes) + self.cfg["roc_eps"])) - 1.0
        oi = safe_div((bids[-1] - asks[-1]), (bids[-1] + asks[-1] + 1e-6))

        # Простейшая оценка тренда: корреляция цены с индексом времени
        t_idx = list(range(len(tail)))
        # ковариация цены и t_idx
        mean_t = rolling_mean(t_idx)
        mean_p = rolling_mean(prices)
        cov_tp = rolling_mean([(ti - mean_t) * (pi - mean_p) for ti, pi in zip(t_idx, prices)])
        var_t = rolling_mean([(ti - mean_t) ** 2 for ti in t_idx]) + 1e-6
        beta = cov_tp / var_t
        trend_strength = bounded(beta, -1.0, 1.0)

        # mean_reversion — сильнее при высоком z и std
        mean_rev = -z * 0.5

        # seasonality — здесь простая заглушка: слабая синусоида по длине окна
        seasonality = math.sin(len(tail) / 12.0) * 0.1

        # liquidity — нормализуем последнее значение в [0,1]
        liq = normalize_01(liquidity[-1], min_val=0.0, max_val=1.0)

        feats = {
            "zscore_momentum": float(z),
            "volatility": float(std_p),
            "roc": float(roc),
            "volume_spike": float(vol_spike),
            "order_imbalance": float(oi),
            "trend_strength": float(trend_strength),
            "mean_reversion": float(mean_rev),
            "seasonality": float(seasonality),
            "liquidity": float(liq),
            "age_minutes": float(HORIZON_TO_MINUTES.get(horizon, 120) / 2),
        }
        if self.cfg.get("normalization", True):
            feats["volatility"] = bounded(feats["volatility"], self.cfg["vol_floor"], 1.0)
        return feats

    @staticmethod
    def _empty() -> Dict[str, float]:
        return {
            "zscore_momentum": 0.0,
            "volatility": 0.0,
            "roc": 0.0,
            "volume_spike": 0.0,
            "order_imbalance": 0.0,
            "trend_strength": 0.0,
            "mean_reversion": 0.0,
            "seasonality": 0.0,
            "liquidity": 0.5,
            "age_minutes": 30.0,
        }

# --------------------------------------------------------------------------------------
# Моделирование: вызов model.py и комбинирование
# --------------------------------------------------------------------------------------

class ModelingPipeline:
    """
    Объединяет расчёты из model.py:
    - базовые вероятности (линейная смесь признаков)
    - байесовское обновление
    - временной декей
    - калибровка (Platt, Isotonic)
    - штрафы за волатильность/ликвидность
    - доверительный интервал
    - ансамбль моделей (stable softmax взвешивание)
    """
    def __init__(self, ensemble_cfg: Dict[str, Any], cal_cfg: Dict[str, Any], risk_cfg: Dict[str, Any]):
        self.ensemble_cfg = ensemble_cfg
        self.cal_cfg = cal_cfg
        self.risk_cfg = risk_cfg
        self.bayes = BayesianUpdater(alpha=5.0, beta=5.0)
        self.tdecay = TemporalDecay()
        self.platt = PlattCalibrator(slope=cal_cfg["platt"]["slope"], bias=cal_cfg["platt"]["bias"])
        self.isotonic = IsotonicCalibrator(bins=cal_cfg["isotonic"]["bins"], min_bin_size=cal_cfg["isotonic"]["min_bin_size"])
        self.ci = ConfidenceInterval(radius=cal_cfg["ci"]["radius"])
        self.vol_pen = VolatilityPenalty(scale=risk_cfg["vol_penalty_scale"])
        self.hmgr = HorizonManager(HORIZON_TO_MINUTES)
        self.ensemble = EnsembleCombiner(temp=ensemble_cfg["softmax_temp"])
        self.risk = RiskManager(config=risk_cfg)

    def base_prob_from_features(self, feats: Dict[str, float]) -> float:
        # Линейная смесь признаков с клипом
        w = self.ensemble_cfg["base_weights"]
        base = (
            0.5
            + w["momentum"] * feats["zscore_momentum"]
            + w["roc"] * feats["roc"]
            + w["volatility"] * feats["volatility"]
            + w["volume_spike"] * feats["volume_spike"]
            + w["order_imbalance"] * feats["order_imbalance"]
            + w["trend_strength"] * feats["trend_strength"]
            + w["mean_reversion"] * feats["mean_reversion"]
            + w["seasonality"] * feats["seasonality"]
            + w["liquidity"] * (feats["liquidity"] - 0.5)
        )
        return bounded(base, *self.ensemble_cfg["prob_clip"])

    def evaluate_horizon(self, feats: Dict[str, float], horizon: str) -> Dict[str, Any]:
        base_prob = self.base_prob_from_features(feats)
        p_bayes, bayes_state = self.bayes.update(base_prob, weight=3.0)
        half_life = self.hmgr.half_life(horizon)
        p_time, recency = self.tdecay.apply(p_bayes, age_minutes=feats["age_minutes"], half_life=half_life)
        p_platt = self.platt.calibrate(p_time)
        p_iso = self.isotonic.calibrate(p_platt)  # порядок: сначала Platt (гладкая), затем Isotonic (монотонная)
        p_vol = self.vol_pen.apply(p_iso, vol=feats["volatility"])
        ci_low, ci_high = self.ci.bounds(p_vol)
        summary = {
            "prob": bounded(p_vol, 0.01, 0.99),
            "ci_low": bounded(ci_low, 0.0, 1.0),
            "ci_high": bounded(ci_high, 0.0, 1.0),
            "recency": recency,
            "bayes": bayes_state,
        }
        # Факторы для диагностики
        factors = {
            "momentum": feats["zscore_momentum"],
            "roc": feats["roc"],
            "volatility": feats["volatility"],
            "volume": feats["volume_spike"],
            "order_imbalance": feats["order_imbalance"],
            "trend_strength": feats["trend_strength"],
            "mean_reversion": feats["mean_reversion"],
            "seasonality": feats["seasonality"],
            "liquidity": feats["liquidity"],
            "recency": recency,
        }
        return {"summary": summary, "factor_weights": factors}

    def combine_horizons(self, outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Комбинируем вероятности разных горизонтов мягким взвешиванием.
        """
        probs = [outputs[h]["summary"]["prob"] for h in outputs]
        weights = stable_softmax(probs, temp=self.ensemble.temp)
        combined_prob = sum(p * w for p, w in zip(probs, weights))
        ci_low = min(outputs[h]["summary"]["ci_low"] for h in outputs)
        ci_high = max(outputs[h]["summary"]["ci_high"] for h in outputs)
        return {
            "prob": bounded(combined_prob, 0.01, 0.99),
            "ci_low": bounded(ci_low, 0.0, 1.0),
            "ci_high": bounded(ci_high, 0.0, 1.0),
            "weights": {h: float(weights[i]) for i, h in enumerate(outputs.keys())},
        }

    def apply_risk(self, prob: float, summary_by_h: Dict[str, Any]) -> float:
        """
        Финальный риск-фильтр: маржа безопасности, пенальти за волу/просадку/низкую уверенность.
        """
        adjusted = self.risk.apply(prob, summary_by_h=summary_by_h)
        return bounded(adjusted, 0.01, 0.99)

# --------------------------------------------------------------------------------------
# Пайплайн: от данных до результатов
# --------------------------------------------------------------------------------------

class EcoPredictPipeline:
    """
    Главный класс: организует полный цикл анализа для одного market_id.
    """

    def __init__(
        self,
        horizons: Optional[List[str]] = None,
        data_fetcher: Optional[DataFetcher] = None,
        preprocessor: Optional[Preprocessor] = None,
        features: Optional[FeatureExtractor] = None,
        modeling: Optional[ModelingPipeline] = None,
    ):
        self.horizons = horizons or DEFAULT_HORIZONS
        self.artifacts = PipelineArtifacts()
        self.fetcher = data_fetcher or DataFetcher(CACHE_CONFIG)
        self.prep = preprocessor or Preprocessor(FEATURE_CONFIG)
        self.fe = features or FeatureExtractor(FEATURE_CONFIG)
        self.model = modeling or ModelingPipeline(ENSEMBLE_CONFIG, CALIBRATION_CONFIG, RISK_CONFIG)

    def run(self, market_id: str) -> Dict[str, Any]:
        t_all = Timer()
        self.artifacts.add_log("info", "pipeline_start", {"market_id": market_id, "horizons": self.horizons})

        try:
            # 1) Загрузка данных
            t = Timer()
            raw = self.fetcher.fetch_market_data(market_id)
            self.artifacts.add_timing("fetch_ms", t.ms())
            self.artifacts.add_input("raw", raw)

            if not raw:
                raise PipelineError("No data fetched", step="fetch", context={"market_id": market_id})

            # 2) Очистка
            t = Timer()
            series = self.prep.clean_series(raw)
            self.artifacts.add_timing("clean_ms", t.ms())
            self.artifacts.add_output("series_clean", series)

            if len(series) < 8:
                self.artifacts.add_log("warn", "too_short_series", {"len": len(series)})

            # 3) Извлечение признаков для каждого горизонта
            t = Timer()
            feats_by_h: Dict[str, Dict[str, float]] = {}
            for h in self.horizons:
                feats_by_h[h] = self.fe.compute_features(series, horizon=h)
            self.artifacts.add_timing("features_ms", t.ms())
            self.artifacts.add_output("features", feats_by_h)

            # 4) Оценка по каждому горизонту
            t = Timer()
            outputs_by_h: Dict[str, Dict[str, Any]] = {}
            top_factors: Dict[str, float] = {}
            for h, feats in feats_by_h.items():
                out = self.model.evaluate_horizon(feats, horizon=h)
                outputs_by_h[h] = out
                # аккумулируем факторные веса (для таба «Факторы»)
                for k, v in out["factor_weights"].items():
                    top_factors[k] = top_factors.get(k, 0.0) + float(v)
            self.artifacts.add_timing("per_horizon_ms", t.ms())
            self.artifacts.add_output("per_horizon", outputs_by_h)
            self.artifacts.add_output("top_factors_raw_sum", top_factors)

            # 5) Комбинация горизонтов (ансамбль)
            t = Timer()
            combined = self.model.combine_horizons(outputs_by_h)
            self.artifacts.add_timing("combine_ms", t.ms())
            self.artifacts.add_output("combined", combined)

            # 6) Риск-фильтр
            t = Timer()
            buy_prob = self.model.apply_risk(combined["prob"], summary_by_h=outputs_by_h)
            self.artifacts.add_timing("risk_ms", t.ms())
            self.artifacts.add_output("buy_prob", buy_prob)

            # 7) Интегральная «сила сигнала»
            signal_strength = self._signal_strength(buy_prob, outputs_by_h)
            self.artifacts.add_output("signal_strength", signal_strength)

            # 8) Метрики качества/диагностики (если есть «истина» или прокси)
            diagnostics = self._diagnostics(series, outputs_by_h, combined, buy_prob)
            self.artifacts.add_output("diagnostics", diagnostics)

            # Финальный словарь для интерфейса
            result = {
                "per_horizon": {h: outputs_by_h[h]["summary"] for h in outputs_by_h},
                "top_factors": top_factors,
                "buy_probs": {h: self.model.apply_risk(outputs_by_h[h]["summary"]["prob"], summary_by_h=outputs_by_h) for h in outputs_by_h},
                "combined": combined,
                "signal_strength": signal_strength,
                "artifacts": self._compact_artifacts(),  # компактный набор для app.py (подробный хранится в self.artifacts)
            }

            self.artifacts.add_log("info", "pipeline_done", {"market_id": market_id})
            self.artifacts.add_timing("total_ms", t_all.ms())
            return result

        except PipelineError as e:
            self.artifacts.add_log("error", "pipeline_error", {"step": e.step, "msg": str(e), "context": e.context})
            self.artifacts.add_timing("total_ms", t_all.ms())
            return self._error_result(f"PipelineError at {e.step}: {e}")

        except Exception as e:
            self.artifacts.add_log("error", "unexpected_error", {"msg": str(e), "trace": traceback.format_exc()})
            self.artifacts.add_timing("total_ms", t_all.ms())
            return self._error_result(f"Unexpected error: {e}")

    # ----------------------------------------------------------------------------------
    # Вспомогательные методы пайплайна
    # ----------------------------------------------------------------------------------

    def _signal_strength(self, buy_prob: float, outputs_by_h: Dict[str, Dict[str, Any]]) -> float:
        """
        Сила сигнала — взвешенная средняя вероятностей покупок по горизонтам и уровню уверенности.
        """
        probs = []
        confs = []
        for h, out in outputs_by_h.items():
            p = out["summary"]["prob"]
            ci_low = out["summary"]["ci_low"]
            ci_high = out["summary"]["ci_high"]
            confidence = max(0.01, ci_high - ci_low)
            probs.append(p)
            confs.append(confidence)
        # взвешивание по confidence
        w = stable_softmax(confs, temp=1.0)
        strength = sum(pp * ww for pp, ww in zip(probs, w))
        # интегрируем общий buy_prob как якорь
        final_strength = 0.6 * buy_prob + 0.4 * strength
        return float(bounded(final_strength, 0.0, 1.0))

    def _diagnostics(
        self,
        series: List[Dict[str, Any]],
        outputs_by_h: Dict[str, Dict[str, Any]],
        combined: Dict[str, Any],
        buy_prob: float,
    ) -> Dict[str, Any]:
        """
        Диагностика: агрегированные статистики ряда и резюме по моделям.
        """
        prices = [x["price"] for x in series]
        volumes = [x["volume"] for x in series]
        vol = rolling_std(prices)
        mean_p = rolling_mean(prices)
        median_p = robust_median(prices)
        min_p, max_p = min(prices), max(prices)

        by_h = {
            h: {
                "prob": out["summary"]["prob"],
                "ci": (out["summary"]["ci_low"], out["summary"]["ci_high"]),
                "recency": out["summary"]["recency"],
            } for h, out in outputs_by_h.items()
        }

        return {
            "series_stats": {
                "mean_price": mean_p,
                "median_price": median_p,
                "min_price": min_p,
                "max_price": max_p,
                "volatility": vol,
                "mean_volume": rolling_mean(volumes),
                "len": len(series),
            },
            "by_horizon": by_h,
            "combined": combined,
            "buy_prob": buy_prob,
        }

    def _compact_artifacts(self) -> Dict[str, Any]:
        """
        Компактная версия артефактов для UI.
        """
        return {
            "logs": self.artifacts.logs[-50:],  # последние 50
            "timings": self.artifacts.timings,
            "metrics": self.artifacts.metrics,
        }

    def _error_result(self, msg: str) -> Dict[str, Any]:
        return {
            "per_horizon": {},
            "top_factors": {},
            "buy_probs": {},
            "combined": {"prob": 0.5, "ci_low": 0.42, "ci_high": 0.58, "weights": {}},
            "signal_strength": 0.0,
            "error": msg,
            "artifacts": self._compact_artifacts(),
        }

# --------------------------------------------------------------------------------------
# Вспомогательные API функции для app.py
# --------------------------------------------------------------------------------------

def run_pipeline(market_id: str, horizons: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Внешняя точка входа для интерфейса.
    """
    pipeline = EcoPredictPipeline(horizons=horizons)
    return pipeline.run(market_id=market_id)

# --------------------------------------------------------------------------------------
# CLI точка входа (опционально)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EcoPredict pipeline runner")
    parser.add_argument("--market", type=str, default="default-market", help="Market ID to analyze")
    parser.add_argument("--horizons", type=str, default="1h,3h,6h,24h", help="Comma-separated horizons")
    args = parser.parse_args()

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    res = run_pipeline(args.market, horizons=horizons)
    print(json.dumps(res, indent=2))
