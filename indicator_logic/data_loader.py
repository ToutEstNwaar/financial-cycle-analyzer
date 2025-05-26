# indicator_logic/data_loader.py

"""
Module for loading and preparing OHLC data from CSV files.
"""

import pandas as pd
import os # Import the os module to check for file existence

def load_ohlc_from_csv(file_path: str) -> pd.DataFrame | None:
  """
  Loads OHLC data from a specified CSV file into a pandas DataFrame.

  Args:
    file_path (str): The path to the CSV file.

  Returns:
    pandas.DataFrame | None: A DataFrame with a DatetimeIndex and columns
                              'open', 'high', 'low', 'close'. Returns None if
                              loading fails, the file doesn't exist, or data
                              is not in the expected format.
  """
  print(f"Attempting to load data from: '{file_path}'") # For logging/debugging

  # Check if the file exists before attempting to read
  if not os.path.exists(file_path):
      print(f"Error: File not found at '{file_path}'")
      return None

  try:
    # Read the CSV file directly using the file path
    # parse_dates=['Date'] tells pandas to try and convert the 'Date' column to datetime objects
    # Using a common date column name 'Date'. If it's different, this might need adjustment
    # or be passed as a parameter.
    df = pd.read_csv(file_path, parse_dates=['Date'])

    # Set the 'Date' column as the index of the DataFrame
    df.set_index('Date', inplace=True)

    # Rename columns to lowercase for consistency and to match expected names
    df.columns = df.columns.str.lower()

    # Define expected OHLC columns
    ohlc_columns = ['open', 'high', 'low', 'close']

    # Check if all required OHLC columns are present
    missing_cols = [col for col in ohlc_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required OHLC columns: {', '.join(missing_cols)}")
        print(f"Available columns: {', '.join(df.columns)}")
        return None

    # Ensure OHLC columns are numeric
    for col in ohlc_columns:
        # errors='coerce' will turn any non-numeric values into NaN (Not a Number)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values in OHLC columns that might result from conversion errors
    # or were already in the file
    df.dropna(subset=ohlc_columns, inplace=True)

    if df.empty:
        print("Error: DataFrame is empty after processing. Check data quality or column names.")
        return None

    # Sort by date to ensure chronological order (important for time series)
    df.sort_index(inplace=True)

    print(f"Data loaded successfully from '{file_path}'.")
    print(f"DataFrame shape: {df.shape}")
    # Optional: Keep these for verbose logging if desired during development
    # print("\n--- First 5 rows: ---")
    # print(df.head())
    # print("\n--- Last 5 rows: ---")
    # print(df.tail())

    return df

  except FileNotFoundError: # Should be caught by os.path.exists, but good to have
    print(f"Error: File not found at '{file_path}' (FileNotFoundError).")
    return None
  except pd.errors.EmptyDataError:
    print(f"Error: The file at '{file_path}' is empty.")
    return None
  except ValueError as ve:
    print(f"Error: Could not parse 'Date' column or other value error in '{file_path}'. Ensure 'Date' column exists and is parsable. Details: {ve}")
    return None
  except Exception as e:
    print(f"An unexpected error occurred while loading or processing the data from '{file_path}': {e}")
    return None

if __name__ == '__main__':
    # Example usage for testing this module directly
    # Create a dummy CSV file for testing
    dummy_data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'Open': [10, 11, 12, 11.5],
        'High': [10.5, 11.2, 12.5, 11.8],
        'Low': [9.8, 10.8, 11.9, 11.2],
        'Close': [10.2, 11.1, 12.2, 11.6],
        'Volume': [1000, 1200, 1500, 1300] # Example of an extra column
    }
    dummy_df = pd.DataFrame(dummy_data)
    test_file_path = 'test_ohlc_data.csv'
    dummy_df.to_csv(test_file_path, index=False)

    print(f"--- Testing load_ohlc_from_csv with '{test_file_path}' ---")
    ohlc_data = load_ohlc_from_csv(test_file_path)

    if ohlc_data is not None:
        print("\nTest load successful. DataFrame info:")
        ohlc_data.info()
        print("\nTest DataFrame head:")
        print(ohlc_data.head())
    else:
        print("\nTest load failed.")

    print(f"\n--- Testing with a non-existent file ---")
    non_existent_data = load_ohlc_from_csv('non_existent_file.csv')
    if non_existent_data is None:
        print("Correctly handled non-existent file.")

    # Clean up dummy file
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"\nCleaned up '{test_file_path}'.")