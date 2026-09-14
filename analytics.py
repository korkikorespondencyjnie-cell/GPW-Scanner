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

def detect_bullish_divergences(pivots: list, rsi_series: pd.Series) -> list:
    """Wykrywa bycze dywergencje na podstawie kolejnych dołków w sekwencji pivotów."""
    lows = [p for p in pivots if p['type'] == 'L']
    divs = []

    for i in range(1, len(lows)):
        p2, p1 = lows[i - 1], lows[i]
        r2 = rsi_series.iloc[p2['idx']]
        r1 = rsi_series.iloc[p1['idx']]
        if p1['price'] < p2['price'] and r1 > r2:
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
        return "Publiczny Udział (Trend Wzrostowy)"
    elif l1 < l2 and h1 < h2:
        return "Trend Spadkowy / Dystrybucja"
    elif l1 > l2 and h1 <= h2:
        return "Akumulacja / Reakumulacja" if vol_diff_pct > 0 else "Korekta Płaska"
    elif l1 <= l2 and h1 > h2:
        return "Dystrybucja / Słabość"
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
    
    div_types = []
    s_0_values = {7: 0.0, 14: 0.0, 28: 0.0}
    days_since_p1 = 0
    p1_price_val, p2_price_val = None, None
    p1_idx_val, p2_idx_val = None, None
    
    dow_pivots = build_strict_dow_pivots(lows, highs, rsi_dict[7], window=WINDOW_PIVOT)
    pivot_lows = [p for p in dow_pivots if p['type'] == 'L']
    pivot_highs = [p for p in dow_pivots if p['type'] == 'H']
    p_l = [p['price'] for p in pivot_lows]
    p_h = [p['price'] for p in pivot_highs]

    for w, rsi_s in rsi_dict.items():
        # Bycza Dywergencja (na podstawie ostatnich dwóch dołków z sekwencji Dowa)
        if len(pivot_lows) >= 2:
            p2_pivot, p1_pivot = pivot_lows[-2], pivot_lows[-1]
            p2, p1 = p2_pivot['price'], p1_pivot['price']
            r2, r1 = rsi_s.iloc[p2_pivot['idx']], rsi_s.iloc[p1_pivot['idx']]
            if p1 < p2 and r1 > r2:
                div_types.append(f"Bycza(RSI{w})")
                dp_pct = ((p1 - p2) / p2) * 100
                dr = r1 - r2
                s_0_values[w] = abs(dp_pct / dr)
                p1_price_val, p2_price_val = p1, p2
                p1_idx_val, p2_idx_val = p1_pivot['idx'], p2_pivot['idx']
                days_since_p1 = len(prices) - 1 - p1_pivot['idx']

        # Niedźwiedzia Dywergencja (na podstawie ostatnich dwóch szczytów z sekwencji Dowa)
        if len(pivot_highs) >= 2:
            p2_pivot, p1_pivot = pivot_highs[-2], pivot_highs[-1]
            p2, p1 = p2_pivot['price'], p1_pivot['price']
            r2, r1 = rsi_s.iloc[p2_pivot['idx']], rsi_s.iloc[p1_pivot['idx']]
            if p1 > p2 and r1 < r2:
                div_types.append(f"Niedźwiedzia(RSI{w})")
                dp_pct = ((p1 - p2) / p2) * 100
                dr = r1 - r2
                s_0_values[w] = -abs(dp_pct / dr)
                if p1_price_val is None:
                    p1_price_val, p2_price_val = p1, p2
                    p1_idx_val, p2_idx_val = p1_pivot['idx'], p2_pivot['idx']
                    days_since_p1 = len(prices) - 1 - p1_pivot['idx']

    # Kalkulacja siły S_D z konfiguracji
    w_rsi = sum(RSI_WEIGHTS[w] * s_0_values[w] for w in RSI_WEIGHTS)
    lambda_t = np.exp(-(np.log(2) / HALF_LIFE_DAYS) * days_since_p1) if days_since_p1 else 0
    s_d = w_rsi * lambda_t
    
    # Wolumen
    v7 = vols.iloc[-7:].mean()
    v90 = vols.iloc[-90:].mean()
    vol_diff_pct = ((v7 - v90) / v90) * 100 if v90 > 0 else 0
    
    current_price = prices.iloc[-1]
    price_change_from_p1 = ((current_price - p1_price_val) / p1_price_val * 100) if p1_price_val else 0.0
    faza_dowa = determine_dow_phase(p_l, p_h, vol_diff_pct)
    ticker_clean = ticker.replace('.WA', '')

    labeled_pivots, pivot_order = label_dow_pivots(dow_pivots)
    ohlc = pd.DataFrame({'High': highs, 'Low': lows, 'Close': prices})
    cons_ratio = calculate_consolidation_ratio(ohlc).iloc[-1]
    cons_ratio_val = round(cons_ratio, 3) if pd.notna(cons_ratio) else None

    summary_row = {
        'Ticker': ticker_clean,
        'Cena': round(current_price, 2),
        'S_D': round(s_d, 4),
        'Kompresja Ratio': cons_ratio_val,
        'Zmiana od P1 %': round(price_change_from_p1, 2),
        'Faza Dowa': faza_dowa,
        'Dywergencje': ", ".join(set(div_types)) if div_types else "Brak",
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
            'days': days_since_p1,
            'change': price_change_from_p1,
            'faza': faza_dowa,
            'kompresja_ratio': cons_ratio_val,
            'pivoty': labeled_pivots,
            'pivot_order': pivot_order,
        }
        
    return summary_row, detailed_report