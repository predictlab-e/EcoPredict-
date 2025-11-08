"""
app.py — расширенный Streamlit интерфейс EcoPredict.

Особенности:
- 10 вкладок: Обзор, Горизонты, Факторы, Дашборд, Корреляции, История, Диагностика, Сырые данные, Метрики, Экспорт
- Расширенные визуализации: Plotly (line, bar, heatmap, gauge, area, scatter), дополнительные графические компоненты
- Интерактив: фильтры горизонтов, выбор факторов, пороги, автообновление, сравнение запусков
- Управление состоянием: история запусков, текущий результат, избранные рынки, предпочтения пользователя
- Обработка ошибок, структурированное логирование, понятные уведомления
- Кэширование на уровне сессии, псевдо-профилировка таймингов
- Гибкий UI с разделами и блоками, пригодный для публикации
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import math
from typing import Dict, Any, List, Optional

# Импорт API верхнего уровня
from Single import run_pipeline

# --------------------------------------------------------------------------------------
# Конфигурация и состояние приложения
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="EcoPredict Dashboard", layout="wide")

# Инициализация session_state
if "history" not in st.session_state:
    st.session_state["history"] = []  # [{"market": str, "timestamp": str, "result": dict}]

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []  # список избранных market_id

if "preferences" not in st.session_state:
    st.session_state["preferences"] = {
        "theme": "light",
        "default_horizons": ["1h", "3h", "6h", "24h"],
        "auto_refresh": False,
        "refresh_interval_sec": 60
    }

if "errors" not in st.session_state:
    st.session_state["errors"] = []  # список ошибок последнего запуска

if "timings" not in st.session_state:
    st.session_state["timings"] = {}  # псевдо-тайминги UI

# --------------------------------------------------------------------------------------
# Боковая панель: управление анализом
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Настройки анализа")
    market_id = st.text_input("Market ID", value="default-market", help="Идентификатор рынка/сигнала для анализа")
    horizons_all = ["1h", "3h", "6h", "24h"]
    horizons = st.multiselect("Horizons", horizons_all, default=st.session_state["preferences"]["default_horizons"])
    st.caption("Выберите горизонты для оценки. Комбинация даёт итоговую вероятность.")
    threshold_buy = st.slider("Порог buy", 0.0, 1.0, 0.55, 0.01)
    auto_refresh = st.checkbox("Автообновление", value=st.session_state["preferences"]["auto_refresh"])
    refresh_interval = st.number_input("Интервал автообновления, сек", min_value=10, max_value=600, value=st.session_state["preferences"]["refresh_interval_sec"])
    run_btn = st.button("🚀 Запустить анализ")
    add_fav_btn = st.button("⭐ Добавить в избранное")
    if add_fav_btn and market_id and market_id not in st.session_state["favorites"]:
        st.session_state["favorites"].append(market_id)
        st.success(f"Добавлено в избранное: {market_id}")

    st.divider()
    st.header("📚 Избранные")
    if st.session_state["favorites"]:
        for m in st.session_state["favorites"]:
            cols = st.columns([4, 1])
            with cols[0]:
                st.write(m)
            with cols[1]:
                if st.button("Анализ", key=f"fav_{m}"):
                    market_id = m
                    st.session_state["last_selected_fav"] = m
                    st.experimental_rerun()
    else:
        st.caption("Добавьте рынки в избранное для быстрого запуска.")

# --------------------------------------------------------------------------------------
# Хелперы UI
# --------------------------------------------------------------------------------------

def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log_error(msg: str, extra: Optional[Dict[str, Any]] = None):
    st.session_state["errors"].append({"t": _timestamp(), "msg": msg, "extra": extra or {}})

def _store_result(market: str, result: Dict[str, Any]):
    st.session_state["last_result"] = result
    st.session_state["history"].append({
        "market": market,
        "timestamp": _timestamp(),
        "result": result
    })

def _safe_df(obj: Any) -> pd.DataFrame:
    try:
        return pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()

def _gauge(value: float, title: str = "Gauge", color: str = "green") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [0, 1]}, "bar": {"color": color}}
    ))
    return fig

def _area(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], fill='tozeroy', mode='lines'))
    fig.update_layout(title=title)
    return fig

def _bar(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    return px.bar(df, x=x, y=y, title=title)

def _line(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None) -> go.Figure:
    return px.line(df, x=x, y=y, color=color, title=title)

# --------------------------------------------------------------------------------------
# Запуск пайплайна
# --------------------------------------------------------------------------------------

def run_and_store(market_id: str, horizons: List[str]):
    t0 = time.time()
    try:
        result = run_pipeline(market_id=market_id, horizons=horizons)
        _store_result(market_id, result)
        st.session_state["timings"]["run_ms"] = (time.time() - t0) * 1000.0
    except Exception as e:
        _log_error("Ошибка пайплайна", {"err": str(e)})
        st.error(f"Ошибка при запуске пайплайна: {e}")

if run_btn:
    run_and_store(market_id, horizons)

if auto_refresh:
    # простое автообновление (перезапуск анализа)
    placeholder = st.empty()
    with placeholder:
        st.info(f"Автообновление каждые {int(refresh_interval)} сек...")
    # НЕ используем бесконечный цикл, чтобы не блокировать UI — обновится при перезагрузке
    st.session_state["preferences"]["auto_refresh"] = True
    st.session_state["preferences"]["refresh_interval_sec"] = int(refresh_interval)

# Получаем текущий результат
result = st.session_state["last_result"]

# --------------------------------------------------------------------------------------
# Верхний заголовок и сводка
# --------------------------------------------------------------------------------------

st.title("EcoPredict — Prediction Market Analytics")

top_cols = st.columns([2, 1, 1, 1])
with top_cols[0]:
    st.markdown(f"**Market:** {market_id}")
with top_cols[1]:
    st.markdown(f"**Horizons:** {', '.join(horizons) if horizons else '—'}")
with top_cols[2]:
    st.markdown(f"**Auto refresh:** {'On' if auto_refresh else 'Off'}")
with top_cols[3]:
    last_run_ms = st.session_state['timings'].get("run_ms", None)
    st.markdown(f"**Run time:** {f'{last_run_ms:.0f} ms' if last_run_ms else '—'}")

st.divider()

# --------------------------------------------------------------------------------------
# Вкладки
# --------------------------------------------------------------------------------------

tabs = st.tabs([
    "📊 Обзор",
    "⏱ Горизонты",
    "📈 Факторы",
    "📟 Дашборд",
    "🧩 Корреляции",
    "🕑 История",
    "🔍 Диагностика",
    "📜 Сырые данные",
    "📏 Метрики",
    "📤 Экспорт"
])

# --------------------------------------------------------------------------------------
# Вкладка 1: Обзор
# --------------------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Интегральная сводка")
    if not result:
        st.warning("Нет данных для отображения. Запустите анализ.")
    else:
        combined = result.get("combined", {})
        signal_strength = float(result.get("signal_strength", 0.0))
        overview_cols = st.columns([1, 1, 2])
        with overview_cols[0]:
            st.metric("Signal Strength", f"{signal_strength:.2f}")
            st.plotly_chart(_gauge(signal_strength, title="Signal strength", color="blue"), use_container_width=True)
        with overview_cols[1]:
            prob = float(combined.get("prob", 0.5))
            st.metric("Combined prob", f"{prob:.2f}")
            st.plotly_chart(_gauge(prob, title="Buy probability", color="green"), use_container_width=True)
        with overview_cols[2]:
            st.write("Weights by horizon")
            w = combined.get("weights", {})
            df_w = _safe_df([{"horizon": k, "weight": v} for k, v in w.items()])
            if not df_w.empty:
                st.plotly_chart(_bar(df_w, x="horizon", y="weight", title="Horizon weights"), use_container_width=True)
            else:
                st.info("Нет данных о весах горизонтов.")

        st.divider()
        st.markdown("### Итоговая рекомендация (эвристическая)")
        decision = "Покупать" if prob >= threshold_buy else "Подождать"
        st.write(f"Порог: {threshold_buy:.2f} → Решение: {decision}")

        st.expander("Комбинированный JSON").write(combined)

# --------------------------------------------------------------------------------------
# Вкладка 2: Горизонты
# --------------------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Результаты по каждому горизонту")
    if not result:
        st.warning("Нет данных.")
    else:
        ph = result.get("per_horizon", {})
        if not ph:
            st.info("Пусто: вероятно пайплайн не вернул результаты по горизонтам.")
        else:
            df = pd.DataFrame(ph).T.reset_index().rename(columns={"index": "horizon"})
            st.dataframe(df, use_container_width=True)

            # Линейный график по вероятностям
            st.plotly_chart(px.line(df, x="horizon", y="prob", title="Probabilities by horizon"), use_container_width=True)

            # Диапазоны доверительных интервалов
            ci_df = pd.DataFrame([{
                "horizon": row["horizon"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"]
            } for _, row in df.iterrows()])
            ci_df["ci_width"] = ci_df["ci_high"] - ci_df["ci_low"]
            st.plotly_chart(px.bar(ci_df, x="horizon", y="ci_width", title="CI width by horizon"), use_container_width=True)

            # Список buy‑prob после риск-фильтра
            buy_probs = result.get("buy_probs", {})
            df_bp = _safe_df([{"horizon": k, "prob": v} for k, v in buy_probs.items()])
            if not df_bp.empty:
                st.plotly_chart(px.bar(df_bp, x="horizon", y="prob", title="Buy prob by horizon (risk-adjusted)"), use_container_width=True)
                st.caption("Вероятности после применения риск‑менеджмента.")
            else:
                st.info("Нет buy_probs.")

# --------------------------------------------------------------------------------------
# Вкладка 3: Факторы
# --------------------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Влияние факторов (суммарно)")
    if not result:
        st.warning("Нет данных.")
    else:
        tf = result.get("top_factors", {})
        if not tf:
            st.info("Нет факторной информации.")
        else:
            df = pd.DataFrame(tf.items(), columns=["factor", "weight"])
            st.dataframe(df, use_container_width=True)

            # Барчарт
            fig_bar = px.bar(df, x="factor", y="weight", title="Factor weights")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Scatter-size/color
            fig_scatter = px.scatter(df, x="factor", y="weight", size="weight", color="weight", title="Factor weights (scatter)")
            fig_scatter.update_layout(xaxis={'categoryorder':'category ascending'})
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Топ‑N факторов
            N = st.slider("Top-N", 3, max(3, len(df)), min(10, len(df)))
            df_top = df.sort_values("weight", ascending=False).head(N)
            st.plotly_chart(px.bar(df_top, x="factor", y="weight", title=f"Top-{N} factors"), use_container_width=True)

# --------------------------------------------------------------------------------------
# Вкладка 4: Дашборд
# --------------------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Композитный дашборд")
    if not result:
        st.warning("Нет данных.")
    else:
        combined = result.get("combined", {})
        ph = result.get("per_horizon", {})
        bp = result.get("buy_probs", {})
        d_cols = st.columns([1, 1, 1])
        with d_cols[0]:
            st.metric("Combined prob", f"{combined.get('prob', 0.5):.2f}")
            st.plotly_chart(_gauge(combined.get("prob", 0.5), "Combined probability", "green"), use_container_width=True)
        with d_cols[1]:
            st.metric("Signal strength", f"{result.get('signal_strength', 0.0):.2f}")
            st.plotly_chart(_gauge(result.get("signal_strength", 0.0), "Signal strength", "blue"), use_container_width=True)
        with d_cols[2]:
            st.metric("Horizons", f"{len(ph)}")
            w = combined.get("weights", {})
            df_w = _safe_df([{"horizon": k, "weight": v} for k, v in w.items()])
            if not df_w.empty:
                st.plotly_chart(_bar(df_w, x="horizon", y="weight", title="Weights"), use_container_width=True)

        st.divider()
        st.markdown("### Buy prob по горизонту")
        df_bp = _safe_df([{"horizon": k, "prob": v} for k, v in bp.items()])
        if not df_bp.empty:
            st.plotly_chart(px.line(df_bp, x="horizon", y="prob", title="Buy prob (risk-adjusted) by horizon"), use_container_width=True)

        st.divider()
        st.markdown("### CI диапазоны")
        df_ph = pd.DataFrame(ph).T.reset_index().rename(columns={"index": "horizon"})
        if not df_ph.empty:
            df_ci = df_ph[["horizon", "ci_low", "ci_high"]].copy()
            df_ci["ci_width"] = df_ci["ci_high"] - df_ci["ci_low"]
            st.plotly_chart(px.bar(df_ci, x="horizon", y="ci_width", title="CI width"), use_container_width=True)

# --------------------------------------------------------------------------------------
# Вкладка 5: Корреляции
# --------------------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Корреляционный анализ факторов (синтетический)")
    if not result:
        st.warning("Нет данных.")
    else:
        tf = result.get("top_factors", {})
        if tf:
            df = pd.DataFrame(tf.items(), columns=["factor", "weight"]).set_index("factor")
            # Синтетическая матрица: корреляция весов с их же ранжированным вариантом
            df["rank"] = df["weight"].rank()
            corr = df.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation heatmap (weights vs ranks)")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Нет факторной информации.")

# --------------------------------------------------------------------------------------
# Вкладка 6: История
# --------------------------------------------------------------------------------------

with tabs[5]:
    st.subheader("История запусков")
    hist = st.session_state["history"]
    if not hist:
        st.info("История пуста.")
    else:
        df_hist = pd.DataFrame([{
            "market": h["market"],
            "timestamp": h["timestamp"],
            "signal": float(h["result"].get("signal_strength", 0.0)),
            "combined_prob": float(h["result"].get("combined", {}).get("prob", 0.5))
        } for h in hist])

        st.dataframe(df_hist, use_container_width=True)
        st.plotly_chart(px.line(df_hist, x="timestamp", y="signal", color="market", title="Signal strength history"), use_container_width=True)
        st.plotly_chart(px.line(df_hist, x="timestamp", y="combined_prob", color="market", title="Combined prob history"), use_container_width=True)

        st.divider()
        st.markdown("### Сравнение двух запусков")
        choices = df_hist["timestamp"].tolist()
        if len(choices) >= 2:
            c1 = st.selectbox("Запуск A", choices, index=len(choices)-2)
            c2 = st.selectbox("Запуск B", choices, index=len(choices)-1)
            runA = next((h for h in hist if h["timestamp"] == c1), None)
            runB = next((h for h in hist if h["timestamp"] == c2), None)
            if runA and runB:
                colA, colB = st.columns(2)
                with colA:
                    st.write("Запуск A:", c1)
                    st.json(runA["result"])
                with colB:
                    st.write("Запуск B:", c2)
                    st.json(runB["result"])
        else:
            st.caption("Нужно минимум 2 запуска для сравнения.")

# --------------------------------------------------------------------------------------
# Вкладка 7: Диагностика
# --------------------------------------------------------------------------------------

with tabs[6]:
    st.subheader("Диагностика пайплайна и ошибки")
    if not result:
        st.warning("Нет данных.")
    else:
        artifacts = result.get("artifacts", {})
        st.markdown("#### Артефакты (компактно)")
        st.json(artifacts)

        st.markdown("#### Последние логи")
        logs = artifacts.get("logs", [])
        df_logs = _safe_df(logs)
        if not df_logs.empty:
            st.dataframe(df_logs.tail(50), use_container_width=True)
        else:
            st.info("Логи отсутствуют.")

        st.markdown("#### Тайминги")
        timings = artifacts.get("timings", {})
        df_timings = _safe_df([{"step": k, "ms": v} for k, v in timings.items()])
        if not df_timings.empty:
            st.plotly_chart(px.bar(df_timings, x="step", y="ms", title="Pipeline timings (ms)"), use_container_width=True)
        else:
            st.info("Тайминги отсутствуют.")

    st.markdown("#### Ошибки UI")
    if st.session_state["errors"]:
        st.write(st.session_state["errors"])
    else:
        st.caption("Ошибок не зарегистрировано.")

# --------------------------------------------------------------------------------------
# Вкладка 8: Сырые данные
# --------------------------------------------------------------------------------------

with tabs[7]:
    st.subheader("Сырые входные данные")
    if not result:
        st.warning("Нет данных.")
    else:
        # В Single.py compact artifacts не содержит inputs; показываем полный result
        st.caption("Компактные артефакты не включают сырые inputs. Если нужен полный фид, проверьте полный JSON результата.")
        st.text_area("Raw JSON", json.dumps(result, indent=2), height=300)

# --------------------------------------------------------------------------------------
# Вкладка 9: Метрики
# --------------------------------------------------------------------------------------

with tabs[8]:
    st.subheader("Метрики качества и статистики серии")
    if not result:
        st.warning("Нет данных.")
    else:
        diagnostics = result.get("diagnostics", {})
        if not diagnostics:
            st.info("Диагностика отсутствует.")
        else:
            st.json(diagnostics)

            stats = diagnostics.get("series_stats", {})
            if stats:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Mean price", f"{float(stats.get('mean_price', 0.0)):.4f}")
                    st.metric("Median price", f"{float(stats.get('median_price', 0.0)):.4f}")
                with cols[1]:
                    st.metric("Volatility", f"{float(stats.get('volatility', 0.0)):.4f}")
                    st.metric("Len", f"{int(stats.get('len', 0))}")
                with cols[2]:
                    st.metric("Mean volume", f"{float(stats.get('mean_volume', 0.0)):.2f}")
                    st.metric("Min/Max", f"{float(stats.get('min_price', 0.0)):.2f} / {float(stats.get('max_price', 0.0)):.2f}")

            by_h = diagnostics.get("by_horizon", {})
            if by_h:
                df_by_h = pd.DataFrame([{
                    "horizon": h,
                    "prob": float(v["prob"]),
                    "ci_low": float(v["ci"][0]),
                    "ci_high": float(v["ci"][1]),
                } for h, v in by_h.items()])
                df_by_h["ci_width"] = df_by_h["ci_high"] - df_by_h["ci_low"]
                st.plotly_chart(px.bar(df_by_h, x="horizon", y="ci_width", title="CI width by horizon (diagnostics)"), use_container_width=True)

# --------------------------------------------------------------------------------------
# Вкладка 10: Экспорт
# --------------------------------------------------------------------------------------

with tabs[9]:
    st.subheader("Экспорт и сохранение")
    if not result:
        st.warning("Нет данных.")
    else:
        # Экспорт текущего результата
        st.download_button(
            label="📥 Скачать текущий результат (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"EcoPredict_{market_id}_{int(time.time())}.json",
            mime="application/json"
        )

        # Экспорт истории
        hist = st.session_state["history"]
        if hist:
            st.download_button(
                label="📥 Скачать историю (JSON)",
                data=json.dumps(hist, indent=2),
                file_name=f"EcoPredict_history_{int(time.time())}.json",
                mime="application/json"
            )
        else:
            st.caption("История пуста — нечего экспортировать.")

        st.divider()
        st.markdown("#### Снимок дашборда (текстовый)")
        snap = {
            "market": market_id,
            "timestamp": _timestamp(),
            "signal_strength": result.get("signal_strength", 0.0),
            "combined_prob": result.get("combined", {}).get("prob", 0.5),
            "decision_threshold": threshold_buy,
            "decision": "buy" if result.get("combined", {}).get("prob", 0.5) >= threshold_buy else "wait",
        }
        st.text_area("Snapshot", json.dumps(snap, indent=2), height=200)

# --------------------------------------------------------------------------------------
# Нижняя панель: подсказки и справка
# --------------------------------------------------------------------------------------

st.divider()
with st.expander("Справка по интерфейсу"):
    st.markdown("""
- Обзор: сводные gauge‑индикаторы, решение по порогу.
- Горизонты: вероятности, доверительные интервалы, риск‑скорректированные buy prob по каждому горизонту.
- Факторы: суммарные факторные веса, топ‑N, scatter/барчарты.
- Дашборд: композитные показатели, веса горизонтов, CI.
- Корреляции: тепловая карта между весами и их рангами (интроспекция).
- История: все запуски, графики изменения сигналов/вероятностей, сравнение двух запусков.
- Диагностика: артефакты пайплайна, логи и тайминги, ошибки UI.
- Сырые данные: полный JSON результата в текстовом виде.
- Метрики: статистики временного ряда, CI ширины по горизонтам.
- Экспорт: выгрузка текущего результата и истории.
""")
