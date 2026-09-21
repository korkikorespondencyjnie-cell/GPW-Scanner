"""
Moduł analityczny: kalkulacja RSI, wyznaczanie dołków/szczytów Dowa,
obliczanie wskaźnika S_D oraz fazy trendu.
"""

import numpy as np
import pandas as pd
from config import RSI_WEIGHTS, HALF_LIFE_DAYS, WINDOW_PIVOT, SD_REPORT_THRESHOLD

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Oblicza Average True Range (ATR) dla pojedynczej spółki (kolumny High, Low, Close)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_consolidation_ratio(df: pd.DataFrame, window_len: int = 5, atr_period: int = 14) -> pd.Series:
    """Kroczący wskaźnik kompresji: (Rolling_Max_High - Rolling_Min_Low) / ATR."""
    atr = calculate_atr(df, period=atr_period)
    rolling_max_h = df['High'].rolling(window=window_len).max()
    rolling_min_l = df['Low'].rolling(window=window_len).min()
    rolling_range = rolling_max_h - rolling_min_l
    return rolling_range / atr

def calc_rsi_series(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_extrema_with_rsi(high_low_series: pd.Series, rsi_series: pd.Series, mode: str = 'low', window: int = 2):
    """
    Wyznacza lokalne ekstremum z otoczeniem RSI.
    Dla ostatnich `window` świec stosuje wyłącznie lookback – bieżąca świeca
    może być dołkiem/szczytem bez potwierdzenia kolejnymi barami.
    """
    extrema_idx, extrema_price, extrema_rsi = [], [], []
    vals = high_low_series.values
    rsi_vals = rsi_series.values
    n = len(vals)

    for i in range(window, n):
        if i + window < n:
            price_slice = vals[i - window : i + window + 1]
            rsi_slice = rsi_vals[max(0, i - window) : min(n, i + window + 1)]
        else:
            price_slice = vals[i - window : i + 1]
            rsi_slice = rsi_vals[max(0, i - window) : i + 1]

        if mode == 'low':
            if vals[i] == min(price_slice):
                extrema_idx.append(i)
                extrema_price.append(vals[i])
                extrema_rsi.append(np.min(rsi_slice))
        elif mode == 'high':
            if vals[i] == max(price_slice):
                extrema_idx.append(i)
                extrema_price.append(vals[i])
                extrema_rsi.append(np.max(rsi_slice))

    return extrema_idx, extrema_price, extrema_rsi

def _merge_extrema_to_pivots(idx_l, p_l, rsi_l, idx_h, p_h, rsi_h) -> list:
    pivots = []
    for i, p, r in zip(idx_l, p_l, rsi_l):
        pivots.append({'idx': i, 'type': 'L', 'price': p, 'rsi': r})
    for i, p, r in zip(idx_h, p_h, rsi_h):
        pivots.append({'idx': i, 'type': 'H', 'price': p, 'rsi': r})
    pivots.sort(key=lambda x: x['idx'])
    return pivots

def build_raw_pivots(lows: pd.Series, highs: pd.Series, rsi_series: pd.Series, window: int = WINDOW_PIVOT) -> list:
    """Zwraca chronologiczną listę wszystkich pivotów L/H bez filtra Dowa."""
    idx_l, p_l, rsi_l = get_extrema_with_rsi(lows, rsi_series, mode='low', window=window)
    idx_h, p_h, rsi_h = get_extrema_with_rsi(highs, rsi_series, mode='high', window=window)
    return _merge_extrema_to_pivots(idx_l, p_l, rsi_l, idx_h, p_h, rsi_h)

def build_strict_dow_pivots(lows: pd.Series, highs: pd.Series, rsi_series: pd.Series, window: int = WINDOW_PIVOT) -> list:
    """Wyznacza sekwencję punktów zwrotnych Dowa z ścisłą naprzemiennością L-H-L-H."""
    pivots = build_raw_pivots(lows, highs, rsi_series, window=window)

    filtered = []
    for p in pivots:
        if not filtered:
            filtered.append(p)
            continue
        last = filtered[-1]
        if last['type'] != p['type']:
            filtered.append(p)
        elif p['type'] == 'L' and p['price'] < last['price']:
            filtered[-1] = p
        elif p['type'] == 'H' and p['price'] > last['price']:
            filtered[-1] = p

    return filtered

def _is_pivot_at_index(series: pd.Series, i: int, window: int, mode: str) -> bool:
    vals = series.values
    n = len(vals)
    if i < window or i >= n:
        return False
    price_slice = vals[i - window : i + window + 1] if i + window < n else vals[i - window : i + 1]
    if mode == 'low':
        return vals[i] == min(price_slice)
    return vals[i] == max(price_slice)

def _empty_current_pivot_state() -> dict:
    return {
        'forming_type': None,
        'forming_idx': None,
        'forming_price': None,
        'is_provisional': False,
    }

def assess_current_pivot_state(
    lows: pd.Series,
    highs: pd.Series,
    window: int = WINDOW_PIVOT,
    last_pivot_type: str | None = None,
) -> dict:
    """
    Interpretuje formujący się pivot na ostatnich świecach (tylko lookback).
    Uwzględnia wyłącznie naprzemienność Dowa: po dołku (L) szukamy szczytu (H) i odwrotnie.
    """
    n = len(lows)
    if n <= window or last_pivot_type not in ('L', 'H'):
        return _empty_current_pivot_state()

    expected_type = 'H' if last_pivot_type == 'L' else 'L'
    mode = 'high' if expected_type == 'H' else 'low'
    series = highs if expected_type == 'H' else lows

    forming = None
    for i in range(n - window, n):
        if not _is_pivot_at_index(series, i, window, mode):
            continue
        forming = {
            'forming_type': expected_type,
            'forming_idx': i,
            'forming_price': float(series.iloc[i]),
            'is_provisional': True,
        }

    return forming if forming else _empty_current_pivot_state()

def _apply_dow_merge(filtered: list, pivot: dict) -> list:
    if not filtered:
        return [pivot]
    last = filtered[-1]
    if last['type'] != pivot['type']:
        filtered.append(pivot)
    elif pivot['type'] == 'L' and pivot['price'] < last['price']:
        filtered[-1] = pivot
    elif pivot['type'] == 'H' and pivot['price'] > last['price']:
        filtered[-1] = pivot
    return filtered

def resolve_analysis_pivots(
    lows: pd.Series,
    highs: pd.Series,
    rsi_series: pd.Series,
    window: int = WINDOW_PIVOT,
) -> dict:
    """
    Łączy pivoty Dowa z interpretacją bieżącego stanu (formujący się L/H)
    do analizy trendu i dywergencji.
    """
    dow_pivots = build_strict_dow_pivots(lows, highs, rsi_series, window=window)
    last_pivot_type = dow_pivots[-1]['type'] if dow_pivots else None
    current = assess_current_pivot_state(lows, highs, window=window, last_pivot_type=last_pivot_type)

    effective_pivots = list(dow_pivots)
    if current['forming_type'] is not None:
        forming_pivot = {
            'idx': current['forming_idx'],
            'type': current['forming_type'],
            'price': current['forming_price'],
            'rsi': float(rsi_series.iloc[current['forming_idx']]),
            'provisional': current['is_provisional'],
        }
        last = effective_pivots[-1] if effective_pivots else None
        if last is None:
            effective_pivots = [forming_pivot]
        elif last['idx'] == forming_pivot['idx'] and last['type'] == forming_pivot['type']:
            effective_pivots[-1] = {**last, **forming_pivot}
        else:
            effective_pivots = _apply_dow_merge(effective_pivots, forming_pivot)

    pivot_lows = [p for p in effective_pivots if p['type'] == 'L']
    pivot_highs = [p for p in effective_pivots if p['type'] == 'H']
    return {
        'dow_pivots': effective_pivots,
        'pivot_lows': pivot_lows,
        'pivot_highs': pivot_highs,
        'current_state': current,
    }

def _calc_divergence_strength(p1: float, p2: float, r1: float, r2: float) -> float:
    """P1/P2 i R1/R2 – pierwszy i drugi pivot chronologicznie."""
    dp_pct = ((p2 - p1) / p1) * 100
    dr = r2 - r1
    return abs(dp_pct / dr)

def _record_divergence_pivot(
    p1_pivot: dict,
    p2_pivot: dict,
    p1: float,
    p2: float,
    prices_len: int,
    p1_price_val,
    p2_price_val,
    p1_idx_val,
    p2_idx_val,
    days_since_p2,
):
    if p1_price_val is not None:
        return p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2
    return p1, p2, p1_pivot['idx'], p2_pivot['idx'], prices_len - 1 - p2_pivot['idx']

def evaluate_divergences(
    pivot_lows: list,
    pivot_highs: list,
    rsi_dict: dict,
    prices_len: int,
) -> tuple[list, dict, float | None, float | None, int | None, int | None, int]:
    """Wykrywa dywergencje klasyczne (B/N) i ukryte (HB/HN) na ostatnich pivotach."""
    div_types = []
    s_0_values = {w: 0.0 for w in RSI_WEIGHTS}
    p1_price_val, p2_price_val = None, None
    p1_idx_val, p2_idx_val = None, None
    days_since_p2 = 0

    for w, rsi_s in rsi_dict.items():
        if len(pivot_lows) >= 2:
            p1_pivot, p2_pivot = pivot_lows[-2], pivot_lows[-1]
            p1, p2 = p1_pivot['price'], p2_pivot['price']
            r1, r2 = rsi_s.iloc[p1_pivot['idx']], rsi_s.iloc[p2_pivot['idx']]
            if p2 < p1 and r2 > r1:
                div_types.append(f'B{w}')
                s_0_values[w] = _calc_divergence_strength(p1, p2, r1, r2)
                p1_price_val, p2_price_val = p1, p2
                p1_idx_val, p2_idx_val = p1_pivot['idx'], p2_pivot['idx']
                days_since_p2 = prices_len - 1 - p2_pivot['idx']
            elif p2 > p1 and r2 < r1:
                div_types.append(f'HB{w}')
                s_0_values[w] = _calc_divergence_strength(p1, p2, r1, r2)
                p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2 = _record_divergence_pivot(
                    p1_pivot, p2_pivot, p1, p2, prices_len,
                    p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2,
                )

        if len(pivot_highs) >= 2:
            p1_pivot, p2_pivot = pivot_highs[-2], pivot_highs[-1]
            p1, p2 = p1_pivot['price'], p2_pivot['price']
            r1, r2 = rsi_s.iloc[p1_pivot['idx']], rsi_s.iloc[p2_pivot['idx']]
            if p2 > p1 and r2 < r1:
                div_types.append(f'N{w}')
                s_0_values[w] = -_calc_divergence_strength(p1, p2, r1, r2)
                p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2 = _record_divergence_pivot(
                    p1_pivot, p2_pivot, p1, p2, prices_len,
                    p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2,
                )
            elif p2 < p1 and r2 > r1:
                div_types.append(f'HN{w}')
                s_0_values[w] = -_calc_divergence_strength(p1, p2, r1, r2)
                p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2 = _record_divergence_pivot(
                    p1_pivot, p2_pivot, p1, p2, prices_len,
                    p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2,
                )

    return div_types, s_0_values, p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2

def scan_all_divergences(
    pivot_lows: list,
    pivot_highs: list,
    rsi_dict: dict,
    dates_index,
) -> list[dict]:
    """Skanuje wszystkie pary pivotów i zwraca listę wykrytych dywergencji (B/HB/N/HN)."""
    records = []

    def append_record(symbol: str, w: int, p1_pivot: dict, p2_pivot: dict, p1: float, p2: float, r1: float, r2: float):
        records.append({
            'symbol': symbol,
            'rsi_window': w,
            'date': dates_index[p2_pivot['idx']],
            'p1_price': round(p1, 2),
            'p2_price': round(p2, 2),
            'rsi_p1': round(float(r1), 2),
            'rsi_p2': round(float(r2), 2),
            'strength': round(_calc_divergence_strength(p1, p2, r1, r2), 4),
        })

    for w, rsi_s in rsi_dict.items():
        for i in range(1, len(pivot_lows)):
            p1_pivot, p2_pivot = pivot_lows[i - 1], pivot_lows[i]
            p1, p2 = p1_pivot['price'], p2_pivot['price']
            r1, r2 = rsi_s.iloc[p1_pivot['idx']], rsi_s.iloc[p2_pivot['idx']]
            if p2 < p1 and r2 > r1:
                append_record(f'b{w}', w, p1_pivot, p2_pivot, p1, p2, r1, r2)
            elif p2 > p1 and r2 < r1:
                append_record(f'hb{w}', w, p1_pivot, p2_pivot, p1, p2, r1, r2)

        for i in range(1, len(pivot_highs)):
            p1_pivot, p2_pivot = pivot_highs[i - 1], pivot_highs[i]
            p1, p2 = p1_pivot['price'], p2_pivot['price']
            r1, r2 = rsi_s.iloc[p1_pivot['idx']], rsi_s.iloc[p2_pivot['idx']]
            if p2 > p1 and r2 < r1:
                append_record(f'n{w}', w, p1_pivot, p2_pivot, p1, p2, r1, r2)
            elif p2 < p1 and r2 > r1:
                append_record(f'hn{w}', w, p1_pivot, p2_pivot, p1, p2, r1, r2)

    return records

def _divergence_label_sort_key(label: str) -> tuple:
    order = {'B': 0, 'HB': 1, 'N': 2, 'HN': 3}
    for prefix in ('HB', 'HN', 'B', 'N'):
        if label.startswith(prefix):
            return (order[prefix], int(label[len(prefix):]))
    return (99, 0)

def format_divergence_labels(div_types: list) -> str:
    if not div_types:
        return "Brak"
    return ", ".join(sorted(set(div_types), key=_divergence_label_sort_key))

def detect_bullish_divergences(pivots: list, rsi_series: pd.Series) -> list:
    """Wykrywa bycze dywergencje na podstawie kolejnych dołków w sekwencji pivotów."""
    lows = [p for p in pivots if p['type'] == 'L']
    divs = []

    for i in range(1, len(lows)):
        p1, p2 = lows[i - 1], lows[i]
        r1 = rsi_series.iloc[p1['idx']]
        r2 = rsi_series.iloc[p2['idx']]
        if p2['price'] < p1['price'] and r2 > r1:
            divs.append({
                'p1_idx': p1['idx'],
                'p1_price': p1['price'],
                'p2_idx': p2['idx'],
                'p2_price': p2['price'],
            })

    return divs

def label_dow_pivots(dow_pivots: list, max_per_type: int = 2) -> tuple[dict, list]:
    """Etykietuje ostatnie N dołków (L1, L2) i szczytów (H1, H2) w kolejności chronologicznej."""
    recent_lows = [p for p in dow_pivots if p['type'] == 'L'][-max_per_type:]
    recent_highs = [p for p in dow_pivots if p['type'] == 'H'][-max_per_type:]

    tagged = []
    for i, p in enumerate(recent_lows, start=1):
        tagged.append((p['idx'], f'L{i}', p['price']))
    for i, p in enumerate(recent_highs, start=1):
        tagged.append((p['idx'], f'H{i}', p['price']))

    tagged.sort(key=lambda x: x[0])
    labeled = {label: round(price, 2) for _, label, price in tagged}
    order = [label for _, label, _ in tagged]
    return labeled, order

def determine_dow_phase(p_l: list, p_h: list, vol_diff_pct: float) -> str:
    if len(p_l) < 2 or len(p_h) < 2:
        return "Brak danych"
    
    l2, l1 = p_l[-2], p_l[-1]
    h2, h1 = p_h[-2], p_h[-1]
    
    if l1 > l2 and h1 > h2: 
        return "wzrost"
    elif l1 < l2 and h1 < h2:
        return "spadek"
    elif l1 > l2 and h1 <= h2:
        return "wzrost NP" 
    elif l1 <= l2 and h1 > h2:
        return "spadek NP"
    return "Konsolidacja"

def analyze_ticker(ticker: str, df_close, df_high, df_low, df_vol):
    """ Przetwarza pojedynczą spółkę i zwraca wynik tabelaryczny oraz ew. raport. """
    prices = df_close[ticker].dropna()
    lows = df_low[ticker].dropna()
    highs = df_high[ticker].dropna()
    vols = df_vol[ticker].dropna()
    
    if len(prices) < 120:
        return None, None
        
    rsi_dict = {w: calc_rsi_series(prices, w) for w in RSI_WEIGHTS.keys()}
    
    analysis = resolve_analysis_pivots(lows, highs, rsi_dict[7], window=WINDOW_PIVOT)
    dow_pivots = analysis['dow_pivots']
    pivot_lows = analysis['pivot_lows']
    pivot_highs = analysis['pivot_highs']
    current_state = analysis['current_state']
    p_l = [p['price'] for p in pivot_lows]
    p_h = [p['price'] for p in pivot_highs]

    div_types, s_0_values, p1_price_val, p2_price_val, p1_idx_val, p2_idx_val, days_since_p2 = evaluate_divergences(
        pivot_lows, pivot_highs, rsi_dict, len(prices)
    )

    # Kalkulacja siły S_D z konfiguracji
    w_rsi = sum(RSI_WEIGHTS[w] * s_0_values[w] for w in RSI_WEIGHTS)
    lambda_t = np.exp(-(np.log(2) / HALF_LIFE_DAYS) * days_since_p2) if days_since_p2 else 0
    s_d = w_rsi * lambda_t
    
    # Wolumen
    v7 = vols.iloc[-7:].mean()
    v90 = vols.iloc[-90:].mean()
    vol_diff_pct = ((v7 - v90) / v90) * 100 if v90 > 0 else 0
    
    current_price = prices.iloc[-1]
    price_change_from_p2 = ((current_price - p2_price_val) / p2_price_val * 100) if p2_price_val else 0.0
    faza_dowa = determine_dow_phase(p_l, p_h, vol_diff_pct)
    ticker_clean = ticker.replace('.WA', '')

    labeled_pivots, pivot_order = label_dow_pivots(dow_pivots)
    ohlc = pd.DataFrame({'High': highs, 'Low': lows, 'Close': prices})
    cons_ratio = calculate_consolidation_ratio(ohlc).iloc[-1]
    cons_ratio_val = round(cons_ratio, 3) if pd.notna(cons_ratio) else None

    forming_type = current_state.get('forming_type')
    stan_biezacy = f"{forming_type}*" if forming_type and current_state.get('is_provisional') else (forming_type or "—")

    summary_row = {
        'Ticker': ticker_clean,
        'Cena': round(current_price, 2),
        'S_D': round(s_d, 4),
        'Stan bieżący': stan_biezacy,
        'Kompresja Ratio': cons_ratio_val,
        'Zmiana od P2 %': round(price_change_from_p2, 2),
        'Faza Dowa': faza_dowa,
        'Dywergencje': format_divergence_labels(div_types),
        'Vol Diff %': round(vol_diff_pct, 1),
    }
    for label in pivot_order:
        summary_row[label] = labeled_pivots[label]

    detailed_report = None
    if abs(s_d) > SD_REPORT_THRESHOLD and p1_idx_val is not None and p2_idx_val is not None:
        detailed_report = {
            'ticker': ticker_clean,
            'sd': s_d,
            'p1': p1_price_val,
            'p2': p2_price_val,
            'days': days_since_p2,
            'change': price_change_from_p2,
            'faza': faza_dowa,
            'stan_biezacy': stan_biezacy,
            'dywergencje': format_divergence_labels(div_types),
            'kompresja_ratio': cons_ratio_val,
            'pivoty': labeled_pivots,
            'pivot_order': pivot_order,
        }
        
    return summary_row, detailed_report