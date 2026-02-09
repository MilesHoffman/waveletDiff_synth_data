"""
Download Stock Data Script

This script allows the user to download stock data for a given ticker and date range.
It handles stock splits by auto-adjusting the data using yfinance.
The data is saved as a CSV file in the 'data/stocks/' directory.
"""

import yfinance as yf
import pandas as pd
import os
import sys
from datetime import datetime

# Global Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'stocks')
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

def get_valid_date(prompt):
    """
    Prompts the user for a date until a valid format (YYYY-MM-DD) is entered.
    """
    while True:
        date_str = input(prompt).strip()
        try:
            # simple validation
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

def download_data(ticker_symbol, start_date, end_date):
    """
    Downloads stock data using yfinance.
    """
    print(f"Downloading data for {ticker_symbol} from {start_date} to {end_date}...")
    
    # Download data with auto_adjust=True to handle splits/dividends
    # This adjusts Open, High, Low, Close
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if df.empty:
        print("No data found for the given range.")
        return None

    # Reset index to make Date a column
    df = df.reset_index()

    # Ensure columns exist (case sensitive check sometimes needed depending on yfinance version)
    # yfinance usually returns: Date, Open, High, Low, Close, Volume
    # If auto_adjust=True, it returns Open, High, Low, Close, Volume.
    
    # Check if 'Date' is in columns, if not it might be index (handled by reset_index)
    # Rename columns to standard Capitalized if necessary (yfinance is usually consistent)
    
    # Select and reorder columns
    # We want Date + COLUMNS
    required_cols = ['Date'] + COLUMNS
    
    # Check for missing columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Try to proceed or fail? 
        # For Volume, sometimes it's missing on indices, but for stocks it should be there.
    
    # Filter and reorder
    final_df = df[required_cols]
    
    return final_df

def main():
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    print("--- Stock Data Downloader ---")
    ticker = input("Enter Stock Ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("Ticker cannot be empty.")
        return

    start_date = get_valid_date("Enter Start Date (YYYY-MM-DD): ")
    end_date = get_valid_date("Enter End Date (YYYY-MM-DD): ")

    # Validate end > start
    if start_date > end_date:
        print("Error: Start date must be before end date.")
        return

    df = download_data(ticker, start_date, end_date)

    if df is not None:
        filename = f"{ticker}_stock_data.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Successfully saved data to: {filepath}")
            print(f"Rows: {len(df)}")
            print(df.head())
        except Exception as e:
            print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
