"""
utils.py — вспомогательные функции и классы EcoPredict.

Содержимое:
- Логирование (structured logger)
- Таймеры и профилировка
- Кэширование: MemoryCache, LRUCache, DiskCache
- HTTP утилиты: http_get_json
- RateLimiter и Retry
- Статистика: rolling_mean, rolling_std, zscore, robust_median, winsorize
- Нормализация: normalize_01, stable_softmax
- Математика: safe_div, pct_change, exp_weight
- Сериализация: serialize_df_like, deserialize_df_like
"""

import os
import time
import json
import math
import random
import pickle
import hashlib
import requests
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Логирование и таймеры
# --------------------------------------------------------------------------------------

def now_utc() -> str:
    """Возвращает текущее время в формате ISO UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

class Logger:
    """Простой структурированный логгер."""
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        print(json.dumps({"t": now_utc(), "level": "INFO", "msg": msg, "extra": extra or {}}))

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        print(json.dumps({"t": now_utc(), "level": "WARN", "msg": msg, "extra": extra or {}}))

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        print(json.dumps({"t": now_utc(), "level": "ERROR", "msg": msg, "extra": extra or {}}))

logger = Logger()

class Timer:
    """Таймер для измерения времени выполнения."""
    def __init__(self):
        self.start = time.time()
    def ms(self) -> float:
        return (time.time() - self.start) * 1000.0

# --------------------------------------------------------------------------------------
# Кэширование
# --------------------------------------------------------------------------------------

class MemoryCache:
    """Простой in-memory кэш с ограничением по количеству элементов."""
    def __init__(self, max_items: int = 64):
        self.max_items = max_items
        self.store: Dict[str, Any] = {}
    def get(self, key: str) -> Any:
        return self.store.get(key)
    def set(self, key: str, value: Any):
        if len(self.store) >= self.max_items:
            self.store.pop(next(iter(self.store)))
        self.store[key] = value

class LRUCache:
    """Кэш с политикой LRU (Least Recently Used)."""
    def __init__(self, max_items: int = 128):
        self.max_items = max_items
        self.store: Dict[str, Any] = {}
        self.order: List[str] = []
    def get(self, key: str) -> Any:
        if key in self.store:
            self.order.remove(key)
            self.order.append(key)
            return self.store[key]
        return None
    def set(self, key: str, value: Any):
        if key in self.store:
            self.order.remove(key)
        elif len(self.store) >= self.max_items:
            oldest = self.order.pop(0)
            self.store.pop(oldest, None)
        self.store[key] = value
        self.order.append(key)

class DiskCache:
    """Кэш на диске (pickle)."""
    def __init__(self, cache_dir: str = "./cache", ttl_seconds: int = 120):
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)
    def _path(self, key: str) -> str:
        h = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.pkl")
    def get(self, key: str) -> Any:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        if time.time() - os.path.getmtime(path) > self.ttl:
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    def set(self, key: str, value: Any):
        path = self._path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)

# --------------------------------------------------------------------------------------
# HTTP утилиты
# --------------------------------------------------------------------------------------

def http_get_json(url: str, timeout: int = 10) -> Dict[str, Any]:
    """GET запрос и возврат JSON."""
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error("http_get_json_failed", {"url": url, "err": str(e)})
        return {}

# --------------------------------------------------------------------------------------
# Ограничители скорости и retry
# --------------------------------------------------------------------------------------

class RateLimiter:
    """Ограничитель скорости: max_calls за per_seconds."""
    def __init__(self, max_calls: int, per_seconds: int):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.calls: List[float] = []
    def consume(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.per_seconds]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.per_seconds - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.calls.append(time.time())

class Retry:
    """Повтор с экспоненциальной паузой."""
    def __init__(self, max_attempts: int = 3, backoff_base: float = 0.7):
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
    def run(self, func, on_error=None):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func()
            except Exception as e:
                if on_error:
                    on_error(e, attempt)
                time.sleep(self.backoff_base * attempt)
        raise RuntimeError("Retry failed")

# --------------------------------------------------------------------------------------
# Статистические функции
# --------------------------------------------------------------------------------------

def rolling_mean(x: List[float]) -> float:
    return sum(x) / len(x) if x else 0.0

def rolling_std(x: List[float]) -> float:
    if not x:
        return 0.0
    m = rolling_mean(x)
    return math.sqrt(sum((xi - m) ** 2 for xi in x) / len(x))

def zscore(x: List[float]) -> List[float]:
    if not x:
        return []
    m = rolling_mean(x)
    s = rolling_std(x) + 1e-6
    return [(xi - m) / s for xi in x]

def robust_median(x: List[float]) -> float:
    if not x:
        return 0.0
    xs = sorted(x)
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    else:
        return 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def winsorize(x: List[float], limits: Tuple[float, float] = (0.01, 0.99)) -> List[float]:
    if not x:
        return []
    xs = sorted(x)
    n = len(xs)
    lo_idx = int(limits[0] * n)
    hi_idx = int(limits[1] * n)
    lo_val = xs[lo_idx]
    hi_val = xs[min(hi_idx, n - 1)]
    return [min(max(xi, lo_val), hi_val) for xi in x]

# --------------------------------------------------------------------------------------
# Нормализация и математика
# --------------------------------------------------------------------------------------
def bounded(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

def normalize_01(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return bounded((x - min_val) / (max_val - min_val + 1e-9), 0.0, 1.0)

def stable_softmax(x: List[float], temp: float = 1.0) -> List[float]:
    if not x:
        return []
    m = max(x)
    exps = [math.exp((xi - m) / temp) for xi in x]
    s = sum(exps)
    return [ei / s for ei in exps]

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def exp_weight(xs: List[float], alpha: float = 0.9) -> List[float]:
    if not xs:
        return []
    result = [xs[0]]
    for i in range(1, len(xs)):
        result.append(alpha * xs[i] + (1 - alpha) * result[-1])
    return result

def pct_change(xs: List[float]) -> List[float]:
    if len(xs) < 2:
        return []
    return [(xs[i] - xs[i-1]) / (xs[i-1] + 1e-9) for i in range(1, len(xs))]

def serialize_df_like(df: Any) -> bytes:
    return pickle.dumps(df)

def deserialize_df_like(data: bytes) -> Any:
    return pickle.loads(data)

