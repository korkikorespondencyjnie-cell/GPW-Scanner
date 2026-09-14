import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# Dodanie głównego katalogu do sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TICKERS, PERIOD_HIST_YFINANCE
from analytics import calculate_consolidation_ratio

RESULTS_TICKERS_CSV = Path(__file__).resolve().parent / 'last_consolidation_tickers.csv'
RESULTS_SUMMARY_CSV = Path(__file__).resolve().parent / 'last_consolidation_summary.csv'


def count_consolidation_episodes(is_consolidation: pd.Series) -> int:
    """ Zlicza unikalne epizody (ciągi dni) spełniania warunku konsolidacji. """
    diffs = is_consolidation.diff()
    episodes = (diffs == 1).sum()
    
    # Jeśli konsolidacja trwała od pierwszego analizowanego dnia
    if len(is_consolidation) > 0 and is_consolidation.iloc[0] == 1:
        episodes += 1
        
    return int(episodes)


def download_portfolio_data(tickers: list[str], period: str) -> pd.DataFrame:
    """
    Pobiera dane masowo przez yf.download() i zapewnia poprawny format tickerów (.WA).
    """
    # GŁÓWNY PUNKT UTRATY DANYCH 1: Sufiksy Giełdy (np. GPW wymaga .WA)
    # Upewniamy się, że każdy ticker ma sufiks .WA
    formatted_tickers = [t if t.endswith(".WA") else f"{t}.WA" for t in tickers]
    
    print(f" Pobieranie danych zbiorczych (yf.download) dla {len(formatted_tickers)} spółek...")
    
    # Pobieramy pełny pakiet danych (High, Low, Close)
    df_raw = yf.download(
        tickers=formatted_tickers,
        period=period,
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )
    return df_raw, formatted_tickers


def process_ticker_from_download(df_raw: pd.DataFrame, ticker_symbol: str, thresholds: list[float], window_len: int, atr_period: int, days_back: int) -> dict | None:
    """
    Wyciąga i przelicza statystyki dla pojedynczego waloru ze zbiorczego DataFrame.
    """
    # GŁÓWNY PUNKT UTRATY DANYCH 2: Obsługa MultiIndexu z yf.download()
    # Gdy pobieramy wiele tickerów, yf.download tworzy kolumny dwupoziomowe: (Ticker, PriceType)
    try:
        if isinstance(df_raw.columns, pd.MultiIndex):
            if ticker_symbol in df_raw.columns.levels[0]:
                df_single = df_raw[ticker_symbol].copy()
            else:
                return None
        else:
            df_single = df_raw.copy()
    except KeyError:
        return None

    # Czyszczenie z ewentualnych dni bez sesji (samych wartości NaN)
    df_single = df_single.dropna(how='all')

    # GŁÓWNY PUNKT UTRATY DANYCH 3: Zbyt rygorystyczny warunek minimalnej długości historii
    # Potrzebujemy min. atr_period + window_len (np. 14 + 5 = 19 świec) do policzenia pierwszego Ratio
    min_required_len = atr_period + window_len
    if df_single.empty or len(df_single) < min_required_len:
        return None

    # Obliczenie współczynnika kompresji za pomocą funkcji w analytics.py
    ratio_series = calculate_consolidation_ratio(df_single, window_len=window_len, atr_period=atr_period)

    # GŁÓWNY PUNKT UTRATY DANYCH 4: Kolejność wycinania okna testowego
    # Obliczamy wskaźniki na pełnej historii, a DOPIERO POTEM odcinamy ostatnie N dni (.tail)
    eval_ratio = ratio_series.tail(days_back).dropna()
    total_eval_days = len(eval_ratio)

    if total_eval_days == 0:
        return None

    clean_name = ticker_symbol.replace('.WA', '')
    ticker_row = {'Ticker': clean_name, 'total_eval_days': total_eval_days}

    # Przypisanie statystyk dla każdego progu
    for th in thresholds:
        is_consolidation = (eval_ratio < th).astype(int)
        active_days = int(is_consolidation.sum())
        episodes = count_consolidation_episodes(is_consolidation)

        pct_time = (active_days / total_eval_days) * 100
        ticker_row[f'th_{th}_days'] = active_days
        ticker_row[f'th_{th}_episodes'] = episodes
        ticker_row[f'th_{th}_str'] = f"{active_days}d ({pct_time:.0f}%)"

    return ticker_row


def run_historical_consolidation_test(window_len: int = 5, atr_period: int = 14, days_back: int = 252):
    """ Główna funkcja wykonująca backtest konsolidacji dla metody download(). """
    df_raw, formatted_tickers = download_portfolio_data(TICKERS, PERIOD_HIST_YFINANCE)

    if df_raw.empty:
        print(" BŁĄD: Nie pobrano żadnych danych z yfinance.")
        return

    thresholds = [round(th, 1) for th in np.arange(0.4, 2.1, 0.2)]
    threshold_total_days = {th: 0 for th in thresholds}
    threshold_episodes_count = {th: 0 for th in thresholds}

    ticker_display_rows = []
    analyzed_tickers_count = 0

    for ticker in formatted_tickers:
        res = process_ticker_from_download(
            df_raw=df_raw,
            ticker_symbol=ticker,
            thresholds=thresholds,
            window_len=window_len,
            atr_period=atr_period,
            days_back=days_back
        )

        if res is None:
            continue

        analyzed_tickers_count += 1
        display_row = {'Ticker': res['Ticker']}

        # Agregacja zbiorcza dla całego portfela
        for th in thresholds:
            threshold_total_days[th] += res[f'th_{th}_days']
            threshold_episodes_count[th] += res[f'th_{th}_episodes']
            display_row[f'th_{th}'] = res[f'th_{th}_str']

        ticker_display_rows.append(display_row)

    if not ticker_display_rows:
        print(" BRAK DANYCH: Żadna spółka nie mogła zostać przetworzona.")
        return

    df_tickers = pd.DataFrame(ticker_display_rows)

    # 1. Tabela dla poszczególnych spółek
    print("\n" + "=" * 110)
    print(f" HISTORYCZNY CZAS TRWANIA KONSOLIDACJI (OSTATNIE {days_back} SESJI / ~1 ROK)")
    print("=" * 110)
    print(" Dni sesyjne w konsolidacji (wraz z udziałem % w roku) dla progów Ratio < X:\n")
    print(df_tickers.to_string(index=False))

    # 2. Tabela Zbiorcza Portfela
    print("\n" + "=" * 80)
    print(" PODSUMOWANIE ZBIORCZE DLA CAŁEGO PORTFELA SPÓŁEK:")
    print("=" * 80)

    total_possible_days = analyzed_tickers_count * days_back
    summary_data = []

    for th in thresholds:
        tot_days = threshold_total_days[th]
        tot_episodes = threshold_episodes_count[th]
        portfolio_pct = (tot_days / total_possible_days) * 100 if total_possible_days > 0 else 0
        avg_episode_len = (tot_days / tot_episodes) if tot_episodes > 0 else 0

        summary_data.append({
            'Próg (Ratio < X)': th,
            'Suma Dni w Konsolidacji': tot_days,
            '% Czasu Portfela': round(portfolio_pct, 2),
            'Liczba Epizodów': tot_episodes,
            'Śr. Długość Epizodu (sesje)': round(avg_episode_len, 2),
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    df_tickers.to_csv(RESULTS_TICKERS_CSV, index=False)
    df_summary.to_csv(RESULTS_SUMMARY_CSV, index=False)
    print(f"\n Wyniki zapisano do:")
    print(f"  • {RESULTS_TICKERS_CSV}")
    print(f"  • {RESULTS_SUMMARY_CSV}")


if __name__ == "__main__":
    run_historical_consolidation_test(window_len=5, atr_period=14, days_back=252)