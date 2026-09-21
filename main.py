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

    base_cols = ['Ticker', 'Cena', 'S_D', 'Stan bieżący', 'Kompresja Ratio', 'Zmiana od P2 %',
                 'Faza Dowa', 'Dywergencje', 'Vol Diff %']
    pivot_cols = [c for c in ('L1', 'H1', 'L2', 'H2') if c in df_res.columns]
    df_res = df_res[base_cols + pivot_cols]

    print("\n" + "="*110)
    print(" TABELA GŁÓWNA: DYWERGENCJE, OSTATNIE PIVOTY DOWA (L1/H1/L2/H2) I KOMPRESJA")
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
            pivot_line = " | ".join(
                f"{label}: {rep['pivoty'][label]:.2f}" for label in rep['pivot_order']
            )
            print(f" • Pivoty Dowa: {pivot_line}")
            print(f" • Stan bieżący: {rep['stan_biezacy']} | Dywergencje: {rep['dywergencje']}")
            print(f" • Punkty zwrotne P1: {rep['p1']:.2f} PLN | P2: {rep['p2']:.2f} PLN")
            print(f" • Kompresja Ratio (ostatnia świeca): {rep['kompresja_ratio']}")
            print(f" • Czas od P2: {rep['days']} sesji | Zmiana ceny od P2: {rep['change']:+.2f}%")
            print(f" • Faza Dowa: {rep['faza']}")
            print("-" * 60)

if __name__ == "__main__":
    main()