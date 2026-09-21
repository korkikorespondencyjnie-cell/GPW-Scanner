"""
Wypisuje wszystkie wykryte dywergencje (B/HB/N/HN) dla portfela z config.py.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TICKERS, PERIOD_HIST_YFINANCE, RSI_WEIGHTS, WINDOW_PIVOT
from analytics import calc_rsi_series, resolve_analysis_pivots, scan_all_divergences

RESULTS_CSV = Path(__file__).resolve().parent / 'last_divergences.csv'


def collect_divergences_for_ticker(ticker: str) -> list[dict]:
    df = yf.Ticker(ticker).history(period=PERIOD_HIST_YFINANCE)
    if df.empty or len(df) < 120:
        return []

    prices = df['Close']
    lows = df['Low']
    highs = df['High']
    rsi_dict = {w: calc_rsi_series(prices, w) for w in RSI_WEIGHTS}

    analysis = resolve_analysis_pivots(lows, highs, rsi_dict[7], window=WINDOW_PIVOT)
    divs = scan_all_divergences(
        analysis['pivot_lows'],
        analysis['pivot_highs'],
        rsi_dict,
        df.index,
    )

    ticker_clean = ticker.replace('.WA', '')
    for d in divs:
        d['ticker'] = ticker_clean
        d['date'] = pd.Timestamp(d['date']).strftime('%Y-%m-%d')

    return divs


def run_divergence_list():
    print(f" Skanowanie dywergencji dla {len(TICKERS)} spółek...")
    all_records = []

    for ticker in TICKERS:
        all_records.extend(collect_divergences_for_ticker(ticker))

    if not all_records:
        print(" Nie wykryto żadnych dywergencji.")
        return

    df = pd.DataFrame(all_records)
    df = df.rename(columns={
        'ticker': 'Ticker',
        'date': 'Data',
        'symbol': 'Symbol',
        'strength': 'Siła',
        'p1_price': 'Cena_P1',
        'p2_price': 'Cena_P2',
        'rsi_p1': 'RSI_P1',
        'rsi_p2': 'RSI_P2',
    })
    df = df[['Ticker', 'Data', 'Symbol', 'Siła', 'Cena_P1', 'Cena_P2', 'RSI_P1', 'RSI_P2']]
    df = df.sort_values(['Data', 'Ticker', 'Symbol'], ascending=[False, True, True])

    print("\n" + "=" * 120)
    print(" WSZYSTKIE WYKRYTE DYWERGENCJE (B / HB / N / HN)")
    print("=" * 120)
    print(df.to_string(index=False))
    print(f"\n Łącznie: {len(df)} dywergencji")

    df.to_csv(RESULTS_CSV, index=False)
    print(f" Wyniki zapisano do: {RESULTS_CSV}")


if __name__ == '__main__':
    run_divergence_list()
