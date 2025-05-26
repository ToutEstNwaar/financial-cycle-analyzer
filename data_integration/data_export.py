"""
Data Export Module for Financial Cycle Analyzer

Handles exporting analysis results and data to various formats:
- CSV export for price data and cycle results
- JSON export for complete analysis results
- Excel export for comprehensive reports
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import os
from io import BytesIO

# Set up logging
logger = logging.getLogger(__name__)


class DataExporter:
    """
    Handles exporting analysis results and data to various formats.
    """
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created export directory: {self.output_dir}")
    
    def export_price_data_csv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Export price data to CSV format.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Path to the exported CSV file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_price_data_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Format the DataFrame for export
            export_df = df.copy()
            if 'Date' in export_df.columns:
                export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            export_df.to_csv(filepath, index=False)
            
            logger.info(f"Exported price data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting price data to CSV: {e}")
            raise
    
    def export_cycle_results_csv(self, cycle_results: List[Dict], symbol: str, timeframe: str) -> str:
        """
        Export cycle analysis results to CSV format.
        
        Args:
            cycle_results: List of detected cycles
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Path to the exported CSV file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_cycle_results_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert to DataFrame
            df = pd.DataFrame(cycle_results)
            
            # Ensure consistent column order
            column_order = ['rank', 'period', 'amplitude', 'phase', 'bartels_score', 'strength']
            df = df.reindex(columns=[col for col in column_order if col in df.columns])
            
            df.to_csv(filepath, index=False)
            
            logger.info(f"Exported cycle results to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting cycle results to CSV: {e}")
            raise
    
    def export_wave_data_csv(self, wave_data: Dict, symbol: str, timeframe: str) -> str:
        """
        Export wave data to CSV format.
        
        Args:
            wave_data: Dictionary with past and future wave data
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            Path to the exported CSV file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol.replace('/', '_')}_{timeframe}_wave_data_{timestamp}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Prepare data for export
            export_data = []
            
            # Add past wave data
            if 'past' in wave_data and wave_data['past']:
                for i, value in enumerate(wave_data['past']):
                    export_data.append({
                        'index': i,
                        'wave_type': 'past',
                        'value': value
                    })
            
            # Add future wave data
            if 'future' in wave_data and wave_data['future']:
                for i, value in enumerate(wave_data['future']):
                    export_data.append({
                        'index': i,
                        'wave_type': 'future',
                        'value': value
                    })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Exported wave data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting wave data to CSV: {e}")
            raise
    
    def export_complete_analysis_json(self, analysis_results: Dict) -> str:
        """
        Export complete analysis results to JSON format.
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            Path to the exported JSON file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = analysis_results.get('symbol', 'unknown').replace('/', '_')
            timeframe = analysis_results.get('timeframe', 'unknown')
            filename = f"{symbol}_{timeframe}_complete_analysis_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Prepare data for JSON serialization
            export_data = self._prepare_for_json_export(analysis_results)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported complete analysis to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting complete analysis to JSON: {e}")
            raise
    
    def export_analysis_summary_json(self, analysis_results: Dict, include_waves: bool = False) -> str:
        """
        Export analysis summary to JSON format (without full wave data).
        
        Args:
            analysis_results: Complete analysis results dictionary
            include_waves: Whether to include wave data in the summary
            
        Returns:
            Path to the exported JSON file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = analysis_results.get('symbol', 'unknown').replace('/', '_')
            timeframe = analysis_results.get('timeframe', 'unknown')
            filename = f"{symbol}_{timeframe}_analysis_summary_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create summary data
            summary_data = {
                'run_id': analysis_results.get('run_id'),
                'symbol': analysis_results.get('symbol'),
                'timeframe': analysis_results.get('timeframe'),
                'analysis_time': analysis_results.get('start_time'),
                'parameters': analysis_results.get('parameters', {}),
                'cycle_count': len(analysis_results.get('cycles', [])),
                'top_cycles': analysis_results.get('cycles', [])[:5],  # Top 5 cycles only
                'status': analysis_results.get('status')
            }
            
            if include_waves:
                waves = analysis_results.get('waves', {})
                summary_data['wave_summary'] = {
                    'past_wave_length': len(waves.get('past', [])),
                    'future_wave_length': len(waves.get('future', [])),
                    'past_wave_range': {
                        'min': min(waves.get('past', [0])),
                        'max': max(waves.get('past', [0]))
                    } if waves.get('past') else None,
                    'future_wave_range': {
                        'min': min(waves.get('future', [0])),
                        'max': max(waves.get('future', [0]))
                    } if waves.get('future') else None
                }
            
            # Prepare for JSON serialization
            export_data = self._prepare_for_json_export(summary_data)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported analysis summary to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting analysis summary to JSON: {e}")
            raise
    
    def export_to_streamlit_download(self, data: Any, filename: str, file_format: str = 'json') -> bytes:
        """
        Prepare data for Streamlit download button.
        
        Args:
            data: Data to export (DataFrame, dict, etc.)
            filename: Name for the file
            file_format: Format for export ('json', 'csv')
            
        Returns:
            Bytes data for download
        """
        try:
            if file_format.lower() == 'csv':
                if isinstance(data, pd.DataFrame):
                    return data.to_csv(index=False).encode('utf-8')
                elif isinstance(data, (list, dict)):
                    df = pd.DataFrame(data)
                    return df.to_csv(index=False).encode('utf-8')
                else:
                    raise ValueError("Data must be DataFrame, list, or dict for CSV export")
            
            elif file_format.lower() == 'json':
                if isinstance(data, pd.DataFrame):
                    json_data = data.to_dict(orient='records')
                else:
                    json_data = data
                
                # Prepare for JSON serialization
                json_data = self._prepare_for_json_export(json_data)
                return json.dumps(json_data, indent=2, default=str).encode('utf-8')
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error preparing data for Streamlit download: {e}")
            raise
    
    def create_comprehensive_report(self, analysis_results: Dict, price_data: pd.DataFrame) -> str:
        """
        Create a comprehensive analysis report with multiple sheets/sections.
        
        Args:
            analysis_results: Complete analysis results
            price_data: Original price data used for analysis
            
        Returns:
            Path to the exported report file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = analysis_results.get('symbol', 'unknown').replace('/', '_')
            timeframe = analysis_results.get('timeframe', 'unknown')
            filename = f"{symbol}_{timeframe}_comprehensive_report_{timestamp}.xlsx"
            filepath = os.path.join(self.output_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Symbol', 'Timeframe', 'Analysis Date', 'Total Cycles Found', 'Data Points', 'Status'],
                    'Value': [
                        analysis_results.get('symbol'),
                        analysis_results.get('timeframe'),
                        analysis_results.get('start_time'),
                        len(analysis_results.get('cycles', [])),
                        len(price_data),
                        analysis_results.get('status')
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Price data sheet
                price_data.to_excel(writer, sheet_name='Price_Data', index=False)
                
                # Cycle results sheet
                if analysis_results.get('cycles'):
                    cycles_df = pd.DataFrame(analysis_results['cycles'])
                    cycles_df.to_excel(writer, sheet_name='Cycle_Results', index=False)
                
                # Parameters sheet
                if analysis_results.get('parameters'):
                    params_data = []
                    for key, value in analysis_results['parameters'].items():
                        params_data.append({'Parameter': key, 'Value': value})
                    pd.DataFrame(params_data).to_excel(writer, sheet_name='Parameters', index=False)
                
                # Wave data sheet (if not too large)
                waves = analysis_results.get('waves', {})
                if waves and (len(waves.get('past', [])) + len(waves.get('future', []))) < 10000:
                    wave_data = []
                    for i, value in enumerate(waves.get('past', [])):
                        wave_data.append({'Index': i, 'Type': 'Past', 'Value': value})
                    for i, value in enumerate(waves.get('future', [])):
                        wave_data.append({'Index': i, 'Type': 'Future', 'Value': value})
                    
                    if wave_data:
                        pd.DataFrame(wave_data).to_excel(writer, sheet_name='Wave_Data', index=False)
            
            logger.info(f"Created comprehensive report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error creating comprehensive report: {e}")
            raise
    
    def _prepare_for_json_export(self, data: Any) -> Any:
        """
        Prepare data for JSON serialization by converting numpy types and datetime objects.
        
        Args:
            data: Data to prepare
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {key: self._prepare_for_json_export(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json_export(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, datetime):
            return data.isoformat()
        elif pd.isna(data):
            return None
        else:
            return data
    
    def get_export_history(self) -> List[Dict]:
        """
        Get list of previously exported files.
        
        Returns:
            List of export file information
        """
        try:
            if not os.path.exists(self.output_dir):
                return []
            
            files = []
            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'modified': datetime.fromtimestamp(stat.st_mtime)
                    })
            
            # Sort by creation time (newest first)
            files.sort(key=lambda x: x['created'], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error getting export history: {e}")
            return []
    
    def cleanup_old_exports(self, days_to_keep: int = 7) -> int:
        """
        Clean up old export files.
        
        Args:
            days_to_keep: Number of days of exports to keep
            
        Returns:
            Number of files deleted
        """
        try:
            if not os.path.exists(self.output_dir):
                return 0
            
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            deleted_count = 0
            
            for filename in os.listdir(self.output_dir):
                filepath = os.path.join(self.output_dir, filename)
                if os.path.isfile(filepath):
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                        logger.info(f"Deleted old export file: {filename}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old exports: {e}")
            return 0 