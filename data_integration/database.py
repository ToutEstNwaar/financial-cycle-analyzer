"""
Database Manager for Financial Cycle Analyzer

Handles storage and retrieval of:
- OHLCV price data
- Analysis results (cycles, waves, parameters)
- Run metadata and configurations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import json
import logging
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Set up logging
logger = logging.getLogger(__name__)

Base = declarative_base()


class PriceData(Base):
    """Table for storing OHLCV price data."""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisRun(Base):
    """Table for storing analysis run metadata."""
    __tablename__ = 'analysis_runs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    parameters = Column(Text, nullable=False)  # JSON string of parameters
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class CycleResults(Base):
    """Table for storing detected cycle results."""
    __tablename__ = 'cycle_results'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), nullable=False)
    cycle_rank = Column(Integer, nullable=False)
    period = Column(Float, nullable=False)
    amplitude = Column(Float, nullable=False)
    phase = Column(Float, nullable=False)
    bartels_score = Column(Float)
    strength = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class WaveData(Base):
    """Table for storing composite wave data."""
    __tablename__ = 'wave_data'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), nullable=False)
    wave_type = Column(String(10), nullable=False)  # 'past' or 'future'
    index_position = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """
    Manages database operations for the Financial Cycle Analyzer.
    Uses SQLite for simplicity and portability.
    """
    
    def __init__(self, db_path: str = "financial_cycle_analyzer.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.engine = None
        self.Session = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database connection and create tables."""
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Create engine
            self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_price_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Store OHLCV price data in the database.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            
            # Clear existing data for this symbol/timeframe combination
            session.query(PriceData).filter_by(symbol=symbol, timeframe=timeframe).delete()
            
            # Insert new data
            for _, row in df.iterrows():
                price_record = PriceData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=row['Date'],
                    open_price=row['Open'],
                    high_price=row['High'],
                    low_price=row['Low'],
                    close_price=row['Close'],
                    volume=row['Volume']
                )
                session.add(price_record)
            
            session.commit()
            session.close()
            
            logger.info(f"Stored {len(df)} price records for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_price_data(self, symbol: str, timeframe: str, 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve price data from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            session = self.Session()
            
            query = session.query(PriceData).filter_by(symbol=symbol, timeframe=timeframe)
            
            if start_date:
                query = query.filter(PriceData.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceData.timestamp <= end_date)
            
            query = query.order_by(PriceData.timestamp)
            
            records = query.all()
            session.close()
            
            if not records:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for record in records:
                data.append({
                    'Date': record.timestamp,
                    'Open': record.open_price,
                    'High': record.high_price,
                    'Low': record.low_price,
                    'Close': record.close_price,
                    'Volume': record.volume
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} price records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving price data: {e}")
            return pd.DataFrame()
    
    def store_analysis_results(self, run_id: str, symbol: str, timeframe: str,
                             parameters: Dict, cycle_results: List[Dict],
                             wave_data: Dict) -> bool:
        """
        Store complete analysis results in the database.
        
        Args:
            run_id: Unique identifier for this analysis run
            symbol: Trading pair symbol
            timeframe: Data timeframe
            parameters: Analysis parameters used
            cycle_results: List of detected cycles
            wave_data: Past and future wave data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            
            # Store analysis run metadata
            analysis_run = AnalysisRun(
                run_id=run_id,
                symbol=symbol,
                timeframe=timeframe,
                parameters=json.dumps(parameters),
                start_time=datetime.utcnow(),
                status='completed'
            )
            session.add(analysis_run)
            
            # Store cycle results
            for i, cycle in enumerate(cycle_results):
                cycle_record = CycleResults(
                    run_id=run_id,
                    cycle_rank=i + 1,
                    period=cycle.get('period', 0),
                    amplitude=cycle.get('amplitude', 0),
                    phase=cycle.get('phase', 0),
                    bartels_score=cycle.get('bartels_score'),
                    strength=cycle.get('strength')
                )
                session.add(cycle_record)
            
            # Store wave data
            for wave_type, wave_values in wave_data.items():
                if wave_values is not None:
                    for i, value in enumerate(wave_values):
                        wave_record = WaveData(
                            run_id=run_id,
                            wave_type=wave_type,
                            index_position=i,
                            value=float(value) if not np.isnan(value) else 0.0
                        )
                        session.add(wave_record)
            
            session.commit()
            session.close()
            
            logger.info(f"Stored analysis results for run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_analysis_results(self, run_id: str) -> Optional[Dict]:
        """
        Retrieve analysis results for a specific run.
        
        Args:
            run_id: Analysis run identifier
            
        Returns:
            Dictionary with analysis results or None if not found
        """
        try:
            session = self.Session()
            
            # Get run metadata
            run = session.query(AnalysisRun).filter_by(run_id=run_id).first()
            if not run:
                session.close()
                return None
            
            # Get cycle results
            cycles = session.query(CycleResults).filter_by(run_id=run_id).order_by(CycleResults.cycle_rank).all()
            
            # Get wave data
            waves = session.query(WaveData).filter_by(run_id=run_id).order_by(WaveData.wave_type, WaveData.index_position).all()
            
            session.close()
            
            # Format results
            cycle_data = []
            for cycle in cycles:
                cycle_data.append({
                    'rank': cycle.cycle_rank,
                    'period': cycle.period,
                    'amplitude': cycle.amplitude,
                    'phase': cycle.phase,
                    'bartels_score': cycle.bartels_score,
                    'strength': cycle.strength
                })
            
            wave_data = {'past': [], 'future': []}
            for wave in waves:
                wave_data[wave.wave_type].append(wave.value)
            
            return {
                'run_id': run_id,
                'symbol': run.symbol,
                'timeframe': run.timeframe,
                'parameters': json.loads(run.parameters),
                'start_time': run.start_time,
                'end_time': run.end_time,
                'status': run.status,
                'cycles': cycle_data,
                'waves': wave_data
            }
            
        except Exception as e:
            logger.error(f"Error retrieving analysis results: {e}")
            return None
    
    def get_recent_runs(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """
        Get list of recent analysis runs.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of runs to return
            
        Returns:
            List of run metadata dictionaries
        """
        try:
            session = self.Session()
            
            query = session.query(AnalysisRun)
            if symbol:
                query = query.filter_by(symbol=symbol)
            
            runs = query.order_by(AnalysisRun.created_at.desc()).limit(limit).all()
            session.close()
            
            result = []
            for run in runs:
                result.append({
                    'run_id': run.run_id,
                    'symbol': run.symbol,
                    'timeframe': run.timeframe,
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'status': run.status,
                    'created_at': run.created_at
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving recent runs: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            session = self.Session()
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old analysis runs and related data
            old_runs = session.query(AnalysisRun).filter(AnalysisRun.created_at < cutoff_date).all()
            old_run_ids = [run.run_id for run in old_runs]
            
            if old_run_ids:
                # Delete related cycle results
                session.query(CycleResults).filter(CycleResults.run_id.in_(old_run_ids)).delete(synchronize_session=False)
                
                # Delete related wave data
                session.query(WaveData).filter(WaveData.run_id.in_(old_run_ids)).delete(synchronize_session=False)
                
                # Delete analysis runs
                session.query(AnalysisRun).filter(AnalysisRun.run_id.in_(old_run_ids)).delete(synchronize_session=False)
            
            session.commit()
            session.close()
            
            logger.info(f"Cleaned up {len(old_run_ids)} old analysis runs")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            session = self.Session()
            
            stats = {
                'price_records': session.query(PriceData).count(),
                'analysis_runs': session.query(AnalysisRun).count(),
                'cycle_results': session.query(CycleResults).count(),
                'wave_records': session.query(WaveData).count(),
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
            }
            
            session.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {} 