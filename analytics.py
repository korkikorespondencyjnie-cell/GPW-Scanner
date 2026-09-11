"""
Moduł analityczny: kalkulacja RSI, wyznaczanie dołków/szczytów Dowa,
obliczanie wskaźnika S_D oraz fazy trendu.
"""

import numpy as np
import pandas as pd
from config import RSI_WEIGHTS, HALF_LIFE_DAYS

def calc_rsi_series(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_extrema_with_rsi(high_low_series: pd.Series, rsi_series: pd.Series, mode: str = 'low', window: int = 2):
    extrema_idx, extrema_price, extrema_rsi = [], [], []
    vals = high_low_series.values
    rsi_vals = rsi_series.values
    
    for i in range(window, len(vals) - window):
        if mode == 'low':
            if vals[i] == min(vals[i-window : i+window+1]):
                extrema_idx.append(i)
                extrema_price.append(vals[i])
                rsi_env = rsi_vals[max(0, i-window) : min(len(rsi_vals), i+window+1)]
                extrema_rsi.append(np.min(rsi_env))
        elif mode == 'high':
            if vals[i] == max(vals[i-window : i+window+1]):
                extrema_idx.append(i)
                extrema_price.append(vals[i])
                rsi_env = rsi_vals[max(0, i-window) : min(len(rsi_vals), i+window+1)]
                extrema_rsi.append(np.max(rsi_env))
                
    return extrema_idx, extrema_price, extrema_rsi

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

def generate_ascii_chart(prices: pd.Series, idx_p2: int, idx_p1: int, p2_val: float, p1_val: float, width: int = 40, height: int = 8) -> str:
    sub_prices = prices.iloc[idx_p2:idx_p1+1].values
    if len(sub_prices) < 2:
        return "Niewystarczająca liczba punktów do wykresu."
    
    min_p, max_p = min(sub_prices), max(sub_prices)
    rng = max_p - min_p if max_p != min_p else 1.0
    
    chart = [[" " for _ in range(width)] for _ in range(height)]
    
    for x in range(width):
        data_idx = int(x * (len(sub_prices) - 1) / (width - 1))
        val = sub_prices[data_idx]
        raw_y = int((val - min_p) / rng * (height - 1))
        y = height - 1 - max(0, min(raw_y, height - 1))
        chart[y][x] = "•"
        
    y_p2 = height - 1 - max(0, min(int((p2_val - min_p) / rng * (height - 1)), height - 1))
    y_p1 = height - 1 - max(0, min(int((p1_val - min_p) / rng * (height - 1)), height - 1))
    
    chart[y_p2][0] = "2"
    chart[y_p1][width - 1] = "1"
    
    lines = ["".join(row) for row in chart]
    lines.append(f"P2: {p2_val:.2f}" + " " * max(1, width - 16) + f"P1: {p1_val:.2f}")
    return "\n".join(lines)

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
    
    idx_l, p_l, _ = get_extrema_with_rsi(lows, rsi_dict[7], mode='low')
    idx_h, p_h, _ = get_extrema_with_rsi(highs, rsi_dict[7], mode='high')
    
    for w, rsi_s in rsi_dict.items():
        # Bycza Dywergencja
        _, pl_w, rl_w = get_extrema_with_rsi(lows, rsi_s, mode='low')
        if len(pl_w) >= 2:
            p2, p1 = pl_w[-2], pl_w[-1]
            r2, r1 = rl_w[-2], rl_w[-1]
            if p1 < p2 and r1 > r2:
                div_types.append(f"Bycza(RSI{w})")
                dp_pct = ((p1 - p2) / p2) * 100
                dr = r1 - r2
                s_0_values[w] = abs(dp_pct / dr)
                p1_price_val, p2_price_val = p1, p2
                p1_idx_val, p2_idx_val = idx_l[-1], idx_l[-2]
                days_since_p1 = len(prices) - 1 - idx_l[-1]
                
        # Niedźwiedzia Dywergencja
        _, ph_w, rh_w = get_extrema_with_rsi(highs, rsi_s, mode='high')
        if len(ph_w) >= 2:
            p2, p1 = ph_w[-2], ph_w[-1]
            r2, r1 = rh_w[-2], rh_w[-1]
            if p1 > p2 and r1 < r2:
                div_types.append(f"Niedźwiedzia(RSI{w})")
                dp_pct = ((p1 - p2) / p2) * 100
                dr = r1 - r2
                s_0_values[w] = -abs(dp_pct / dr)
                if p1_price_val is None:
                    p1_price_val, p2_price_val = p1, p2
                    p1_idx_val, p2_idx_val = idx_h[-1], idx_h[-2]
                    days_since_p1 = len(prices) - 1 - idx_h[-1]

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

    summary_row = {
        'Ticker': ticker_clean,
        'Cena': round(current_price, 2),
        'S_D': round(s_d, 4),
        'Zmiana od P1 %': round(price_change_from_p1, 2),
        'Faza Dowa': faza_dowa,
        'Dywergencje': ", ".join(set(div_types)) if div_types else "Brak",
        'Vol Diff %': round(vol_diff_pct, 1)
    }
    
    detailed_report = None
    if abs(s_d) > 0.5 and p1_idx_val is not None and p2_idx_val is not None:
        detailed_report = {
            'ticker': ticker_clean,
            'sd': s_d,
            'p1': p1_price_val,
            'p2': p2_price_val,
            'days': days_since_p1,
            'change': price_change_from_p1,
            'faza': faza_dowa,
            'chart': generate_ascii_chart(prices, p2_idx_val, p1_idx_val, p2_price_val, p1_price_val)
        }
        
    return summary_row, detailed_report