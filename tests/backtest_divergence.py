"""
Backtest byczych dywergencji RSI – symulacja pozycji i raportowanie wyników.
Logika wykrywania pivotów i dywergencji pochodzi z modułu analytics.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TICKERS, PERIOD_HIST_YFINANCE, WINDOW_PIVOT
from analytics import (
    calc_rsi_series,
    build_raw_pivots,
    build_strict_dow_pivots,
    detect_bullish_divergences,
)

BACKTEST_RSI_WINDOW = 14
RESULTS_CSV = Path(__file__).resolve().parent / 'last_backtest_results.csv'


def calculate_swing_metrics_to_first_pivot(divs, pivots, dates_index):
    """
    Oblicza zasięg swingu do najbliższego pierwszego ekstremum po P2
    oraz prostą rentowność w skali roku (Annualized ROR).
    """
    gains = []
    annualized_rors = []

    for d in divs:
        p2_idx = d['p2_idx']
        next_pivots = [p for p in pivots if p['idx'] > p2_idx]
        if not next_pivots:
            continue

        target_pivot = next_pivots[0]
        price_entry = d['p2_price']
        price_exit = target_pivot['price']
        gain_pct = ((price_exit - price_entry) / price_entry) * 100

        p2_date = dates_index[p2_idx]
        exit_date = dates_index[target_pivot['idx']]
        days_held = max(1, (exit_date - p2_date).days)
        annualized_ror = gain_pct * (365.0 / days_held)

        gains.append(gain_pct)
        annualized_rors.append(annualized_ror)

    return gains, annualized_rors


def run_backtest():
    print(" Pobieranie danych do Backtestu...")
    df_close = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['Close']
    df_high = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['High']
    df_low = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['Low']

    stats = []

    for ticker in TICKERS:
        prices = df_close[ticker].dropna()
        highs = df_high[ticker].dropna()
        lows = df_low[ticker].dropna()

        if len(prices) < 120:
            continue

        rsi = calc_rsi_series(prices, BACKTEST_RSI_WINDOW)

        raw_pivots = build_raw_pivots(lows, highs, rsi, window=WINDOW_PIVOT)
        dow_pivots = build_strict_dow_pivots(lows, highs, rsi, window=WINDOW_PIVOT)

        divs_early = detect_bullish_divergences(raw_pivots, rsi)
        divs_dow = detect_bullish_divergences(dow_pivots, rsi)

        gains_e, rors_e = calculate_swing_metrics_to_first_pivot(divs_early, raw_pivots, prices.index)
        gains_d, rors_d = calculate_swing_metrics_to_first_pivot(divs_dow, dow_pivots, prices.index)

        stats.append({
            'Ticker': ticker.replace('.WA', ''),
            'Ilość Wczesne': len(divs_early),
            'Śr. Swing Wczesny %': round(np.mean(gains_e), 2) if gains_e else 0.0,
            'Roczna Rent. Wczesny %': round(np.mean(rors_e), 2) if rors_e else 0.0,
            'Ilość Dow': len(divs_dow),
            'Śr. Swing Dow %': round(np.mean(gains_d), 2) if gains_d else 0.0,
            'Roczna Rent. Dow %': round(np.mean(rors_d), 2) if rors_d else 0.0,
        })

    df_stats = pd.DataFrame(stats)

    print("\n" + "=" * 115)
    print(" PORÓWNANIE: ZAMKNIĘCIE NA PIERWSZYM EXTREMUM (SZCZYT/DOŁEK) + RENTOWNOŚĆ ROCZNA")
    print("=" * 115)
    print(df_stats.to_string(index=False))

    print("\n Podsumowanie Zbiorcze:")
    print(f" • Wczesne: Średni Zasięg = {df_stats['Śr. Swing Wczesny %'].mean():.2f}% | "
          f"Średnia Rentowność Roczna = {df_stats['Roczna Rent. Wczesny %'].mean():.2f}%")
    print(f" • Dowa:    Średni Zasięg = {df_stats['Śr. Swing Dow %'].mean():.2f}% | "
          f"Średnia Rentowność Roczna = {df_stats['Roczna Rent. Dow %'].mean():.2f}%")

    df_stats.to_csv(RESULTS_CSV, index=False)
    print(f"\n Wyniki zapisano do: {RESULTS_CSV}")


if __name__ == "__main__":
    run_backtest()
