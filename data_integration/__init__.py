"""
Data Integration Module for Financial Cycle Analyzer

This module handles:
- Live data fetching from crypto exchanges via CCXT
- Database storage and retrieval
- Data export functionality
"""

from .api_client import CryptoDataFetcher
from .database import DatabaseManager
from .data_export import DataExporter

__all__ = ['CryptoDataFetcher', 'DatabaseManager', 'DataExporter'] 