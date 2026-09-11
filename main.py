"""
Główny skrypt wykonawczy.
Wyciszony, czytelny punkt startowy skanera.
"""

import yfinance as yf
import pandas as pd
from config import TICKERS, PERIOD_HIST_YFINANCE, SD_REPORT_THRESHOLD
from analytics import analyze_ticker

def main():
    print(" Pobieranie danych giełdowych...")
    df_close = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['Close']
    df_high = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['High']
    df_low = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['Low']
    df_vol = yf.download(TICKERS, period=PERIOD_HIST_YFINANCE, multi_level_index=False)['Volume']

    results, reports = [], []

    for ticker in TICKERS:
        row, report = analyze_ticker(ticker, df_close, df_high, df_low, df_vol)
        if row:
            results.append(row)
        if report:
            reports.append(report)

    df_res = pd.DataFrame(results).sort_values(by='S_D', ascending=False)

    print("\n" + "="*110)
    print(" TABELA GŁÓWNA: ANALIZA DYWERGENCJI I FAZY TRENDU DOWA")
    print("="*110)
    print(df_res.to_string(index=False))

    print("\n" + "="*110)
    print(f" SZCZEGÓŁOWE RAPORTY DLA SPÓŁEK Z SIŁĄ DYWERGENCJI |S_D| > {SD_REPORT_THRESHOLD}")
    print("="*110)

    if not reports:
        print("Brak spółek spełniających kryterium siły w bieżącym oknie czasowym.")
    else:
        for rep in reports:
            print(f"\n--- RAPORT SZCZEGÓŁOWY: {rep['ticker']} (S_D = {rep['sd']:.4f}) ---")
            print(f" • Punkt Zwrotny P2: {rep['p2']:.2f} PLN | P1: {rep['p1']:.2f} PLN")
            print(f" • Czas od P1: {rep['days']} sesji | Zmiana ceny od P1: {rep['change']:+.2f}%")
            print(f" • Faza Dowa: {rep['faza']}")
            print("-" * 60)

if __name__ == "__main__":
    main()