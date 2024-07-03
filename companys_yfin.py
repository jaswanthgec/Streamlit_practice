import yfinance as yf
import streamlit as st
import pandas as pd

# Title of the app
st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and ***volume*** of the selected company!
""")

# Dropdown for selecting the company
tickerSymbol = st.selectbox(
    'Select a company',
    ('GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA')
)

# Default period options
period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
period = st.selectbox('Select the period', period_options, index=period_options.index('1d'))

# Get the minimum and maximum dates for the selected ticker
tickerData = yf.Ticker(tickerSymbol)
tickerHistory = tickerData.history(period='max')
min_date = pd.to_datetime(tickerHistory.index.min()).date()
max_date = pd.to_datetime(tickerHistory.index.max()).date()

# Date inputs for start and end date
start_date = st.date_input('Start date', min_date)
end_date = st.date_input('End date', max_date)

# Dropdown for selecting the attribute
attribute = st.selectbox(
    'Select an attribute',
    ('Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits')
)

# Fetch the historical prices for this ticker within the selected date range
tickerDf = tickerData.history(period=period, start=start_date, end=end_date)

# Display the selected attribute
st.write(f"""
## {attribute} of {tickerSymbol}
""")
st.line_chart(tickerDf[attribute])