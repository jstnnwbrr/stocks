import datetime
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Get the S&P 500 list from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
tables = pd.read_html(url)

# The first table usually contains the S&P 500 constituents
sp500_table = tables[0]

# Extract the 'Symbol' column
sp500_tickers = sp500_table['Symbol'].tolist()

# Optional: Sort and remove any non-string entries (some tables might contain weird footnotes)
sp500_tickers = sorted([str(ticker).strip().upper() for ticker in sp500_tickers if isinstance(ticker, str)])

# Output the list
print(sp500_tickers)

def get_data(stock_name, end_date):
    df = yf.download(stock_name, start='2008-01-01', end=end_date, multi_level_index=False)
    df = df.reset_index()
    df = df.set_index('Date').asfreq('B').dropna()
    df['Close'].plot(title=f"{stock_name}")
    plt.show()
    
    return df

today = datetime.date.today()
end_date = today + pd.offsets.BusinessDay(1)

for stock_name in sp500_tickers:
    df = get_data(stock_name, end_date)
    print("\n", stock_name, "\nClose: ", df['Close'].iloc[-1], "\n\n")
