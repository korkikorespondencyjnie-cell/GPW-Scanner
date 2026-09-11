"""
Plik konfiguracyjny skanera dywergencji.
Zawiera listy tickerów oraz stałe wagowe i skali.
"""

RAW_TICKERS = [
    'ZAB', 'DNP', 'ALE', 'PCO', 'PGE', 'TPE', 'JSW', 'OPL', 'MIL', 'ATT', 
    'CPS', 'ENA', 'PXM', 'APR', 'LBW', 'TXT', 'BMC', 'COG', 'LWB', 'TOR', 
    'EUR', 'MRB', 'ICE', 'DVL', 'ELT', 'GPP', '1AT', 'KGN', 'GRX'
]

TICKERS = [t + '.WA' for t in RAW_TICKERS]

# Parametry analizy technicznej
WINDOW_PIVOT = 2             # Liczba świec przed i po dołku/szczycie
PERIOD_HIST_YFINANCE = '1y'  # Zakres danych z Yahoo Finance
HALF_LIFE_DAYS = 10          # Półokres zaniku czasowego dla dywergencji (w dniach)

# Wagi dla poszczególnych horyzontów RSI w kalkulacji siły S_D
RSI_WEIGHTS = {
    7: 0.15,
    14: 0.35,
    28: 0.50
}

# Progi raportowania
SD_REPORT_THRESHOLD = 0.5    # Minimalna siła dywergencji dla raportu szczegółowego