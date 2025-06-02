"""
Cryptocurrency Data Fetcher using CCXT

Handles fetching OHLCV data from various exchanges for BTC/USD pairs.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from typing import Optional, Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataFetcher:
    """
    Fetches cryptocurrency OHLCV data from exchanges using CCXT.
    Currently focused on BTC/USD pairs with multiple timeframe support.
    """
    
    # Supported timeframes
    TIMEFRAMES = {
        '1m': '1m',
        '5m': '5m', 
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w'
    }
    
    # Supported exchanges (prioritized by reliability)
    EXCHANGES = {
        'binance': ccxt.binance,
        'coinbase': ccxt.coinbase,
        'kraken': ccxt.kraken,
        'bybit': ccxt.bybit
    }
    
    def __init__(self, exchange_name: str = 'binance', sandbox: bool = False):
        """
        Initialize the data fetcher.
        
        Args:
            exchange_name: Name of the exchange to use
            sandbox: Whether to use sandbox/testnet mode
        """
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        self.exchange = None
        self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize the CCXT exchange instance."""
        try:
            if self.exchange_name not in self.EXCHANGES:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
                
            exchange_class = self.EXCHANGES[self.exchange_name]
            self.exchange = exchange_class({
                'sandbox': self.sandbox,
                'rateLimit': 1200,  # Be respectful to APIs
                'enableRateLimit': True,
            })
            
            # Load markets
            self.exchange.load_markets()
            logger.info(f"Successfully initialized {self.exchange_name} exchange")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {self.exchange_name}: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available BTC trading pairs."""
        try:
            markets = self.exchange.markets
            btc_symbols = [symbol for symbol in markets.keys() 
                          if 'BTC' in symbol and ('USD' in symbol or 'USDT' in symbol)]
            return sorted(btc_symbols)
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return ['BTC/USD', 'BTC/USDT']  # Fallback
    
    def fetch_ohlcv(self, 
                    symbol: str = 'BTC/USD', 
                    timeframe: str = '1d',
                    limit: int = 1000,
                    since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe for the data
            limit: Maximum number of candles to fetch
            since: Start date for fetching data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Validate inputs
            if timeframe not in self.TIMEFRAMES:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Convert since to timestamp if provided
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            # Fetch data
            logger.info(f"Fetching {symbol} {timeframe} data (limit: {limit})")
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )
            
            if not ohlcv:
                raise ValueError("No data returned from exchange")
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Convert timestamp to datetime with UTC timezone
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.drop('timestamp', axis=1)
            
            # Reorder columns to match expected format
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            raise
    
    def fetch_latest_price(self, symbol: str = 'BTC/USD') -> Dict:
        """
        Fetch the latest price information for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with latest price info
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000, tz=pytz.utc),
                'change_24h': ticker['percentage']
            }
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            raise
    
    def get_historical_data(self, 
                           symbol: str = 'BTC/USD',
                           timeframe: str = '1d',
                           days_back: int = 365) -> pd.DataFrame:
        """
        Fetch historical data for a specified number of days back.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the data
            days_back: Number of days to fetch back from current date
            
        Returns:
            DataFrame with historical OHLCV data
        """
        try:
            # Calculate start date (use UTC for consistency)
            end_date = datetime.now(pytz.utc)
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch data in chunks if needed (some exchanges have limits)
            all_data = []
            current_date = start_date
            chunk_size = 1000  # Most exchanges support this
            
            while current_date < end_date:
                try:
                    chunk_data = self.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=chunk_size,
                        since=current_date
                    )
                    
                    if chunk_data.empty:
                        break
                        
                    all_data.append(chunk_data)
                    
                    # Update current_date to the last timestamp + 1 period
                    last_date = chunk_data['Date'].iloc[-1]
                    current_date = last_date + pd.Timedelta(self._get_timedelta_from_timeframe(timeframe))
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching chunk starting from {current_date}: {e}")
                    break
            
            if not all_data:
                raise ValueError("No historical data could be fetched")
            
            # Combine all chunks
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            
            # Filter to exact date range
            combined_df = combined_df[
                (combined_df['Date'] >= start_date) & 
                (combined_df['Date'] <= end_date)
            ]
            
            logger.info(f"Fetched {len(combined_df)} historical candles for {symbol}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def _get_timedelta_from_timeframe(self, timeframe: str) -> pd.Timedelta:
        """Convert timeframe string to pandas Timedelta."""
        timeframe_map = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1),
            '1w': pd.Timedelta(weeks=1)
        }
        return timeframe_map.get(timeframe, pd.Timedelta(days=1))
    
    def test_connection(self) -> bool:
        """Test if the exchange connection is working."""
        try:
            # Try to fetch a simple ticker
            self.exchange.fetch_ticker('BTC/USDT')
            logger.info(f"Connection test successful for {self.exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.exchange_name}: {e}")
            return False


# Utility function for easy access
def get_btc_data(timeframe: str = '1d', days_back: int = 365, exchange: str = 'binance') -> pd.DataFrame:
    """
    Convenience function to quickly fetch BTC/USD data.
    
    Args:
        timeframe: Data timeframe
        days_back: Number of days to fetch
        exchange: Exchange to use
        
    Returns:
        DataFrame with BTC OHLCV data
    """
    fetcher = CryptoDataFetcher(exchange_name=exchange)
    
    # Try BTC/USD first, fallback to BTC/USDT
    try:
        return fetcher.get_historical_data('BTC/USD', timeframe, days_back)
    except:
        try:
            return fetcher.get_historical_data('BTC/USDT', timeframe, days_back)
        except Exception as e:
            logger.error(f"Failed to fetch BTC data: {e}")
            raise 