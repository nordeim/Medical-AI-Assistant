"""
Data Management System for Analytics Platform
Handles data ingestion, processing, storage, and retrieval
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class DataSource(Enum):
    DATABASE = "database"
    API = "api"
    FILE = "file"
    STREAM = "stream"
    WEBHOOK = "webhook"

class DataQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    source_id: str
    source_type: DataSource
    connection_string: str
    schedule: str  # cron expression
    enabled: bool = True
    last_updated: Optional[datetime] = None
    retry_attempts: int = 3
    timeout_seconds: int = 300

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    dataset_id: str
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    validity_score: float
    overall_quality: DataQuality
    issues: List[str]
    recommendations: List[str]
    assessment_date: datetime

@dataclass
class DataSchema:
    """Data schema definition"""
    table_name: str
    columns: Dict[str, str]  # column_name: data_type
    primary_key: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[str]
    constraints: List[str]

class DataManager:
    """Advanced Data Management System"""
    
    def __init__(self, db_path: str = "analytics_data.db"):
        self.db_path = db_path
        self.data_sources = {}
        self.schemas = {}
        self.ingestion_configs = {}
        self.data_caches = {}
        self.quality_reports = {}
        
        self._initialize_database()
        self._setup_logging()
    
    def _initialize_database(self) -> None:
        """Initialize the analytics database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create core tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_sources (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    connection_string TEXT,
                    schedule TEXT,
                    enabled BOOLEAN DEFAULT 1,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    table_name TEXT,
                    record_count INTEGER,
                    data_quality TEXT,
                    last_updated TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES data_sources (source_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_results (
                    result_id TEXT PRIMARY KEY,
                    analysis_type TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    parameters TEXT,
                    execution_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_reports (
                    report_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    completeness REAL,
                    accuracy REAL,
                    consistency REAL,
                    timeliness REAL,
                    validity REAL,
                    overall_quality TEXT,
                    issues TEXT,
                    recommendations TEXT,
                    assessment_date TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            logging.info("Database initialized successfully")
            
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def register_data_source(self, config: DataIngestionConfig) -> None:
        """Register a new data source"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_sources 
                (source_id, source_type, connection_string, schedule, enabled, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                config.source_id,
                config.source_type.value,
                config.connection_string,
                config.schedule,
                config.enabled,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.data_sources[config.source_id] = config
            logging.info(f"Data source {config.source_id} registered successfully")
            
        except Exception as e:
            logging.error(f"Failed to register data source {config.source_id}: {e}")
            raise
    
    def ingest_data(self, source_id: str, data: pd.DataFrame) -> str:
        """Ingest data from a registered source"""
        try:
            dataset_id = f"dataset_{source_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store data in database
            self._store_dataframe(data, dataset_id)
            
            # Perform quality assessment
            quality_report = self._assess_data_quality(data, dataset_id)
            
            # Update dataset record
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO datasets
                (dataset_id, source_id, table_name, record_count, data_quality, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                source_id,
                dataset_id,
                len(data),
                quality_report.overall_quality.value,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            # Store quality report
            self.quality_reports[dataset_id] = quality_report
            
            logging.info(f"Data ingested successfully. Dataset ID: {dataset_id}")
            return dataset_id
            
        except Exception as e:
            logging.error(f"Data ingestion failed: {e}")
            raise
    
    def _store_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Store DataFrame in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            logging.info(f"DataFrame stored as table: {table_name}")
        except Exception as e:
            logging.error(f"Failed to store DataFrame: {e}")
            raise
    
    def retrieve_data(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve data from dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name=?
            """, (dataset_id,))
            
            if cursor.fetchone():
                df = pd.read_sql_query(f"SELECT * FROM {dataset_id}", conn)
                conn.close()
                return df
            else:
                conn.close()
                logging.warning(f"Dataset {dataset_id} not found")
                return None
                
        except Exception as e:
            logging.error(f"Failed to retrieve data: {e}")
            return None
    
    def _assess_data_quality(self, data: pd.DataFrame, dataset_id: str) -> DataQualityReport:
        """Assess data quality"""
        try:
            # Calculate quality metrics
            completeness = self._calculate_completeness(data)
            accuracy = self._calculate_accuracy(data)
            consistency = self._calculate_consistency(data)
            timeliness = self._calculate_timeliness(data)
            validity = self._calculate_validity(data)
            
            # Calculate overall quality
            overall_score = (completeness + accuracy + consistency + timeliness + validity) / 5
            
            if overall_score >= 0.9:
                overall_quality = DataQuality.EXCELLENT
            elif overall_score >= 0.8:
                overall_quality = DataQuality.GOOD
            elif overall_score >= 0.7:
                overall_quality = DataQuality.FAIR
            elif overall_score >= 0.6:
                overall_quality = DataQuality.POOR
            else:
                overall_quality = DataQuality.UNACCEPTABLE
            
            # Identify issues
            issues = self._identify_data_issues(data, completeness, accuracy, consistency, timeliness, validity)
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(completeness, accuracy, consistency, timeliness, validity)
            
            # Store quality report
            self._store_quality_report(dataset_id, completeness, accuracy, consistency, timeliness, validity,
                                     overall_quality, issues, recommendations)
            
            return DataQualityReport(
                dataset_id=dataset_id,
                completeness_score=completeness,
                accuracy_score=accuracy,
                consistency_score=consistency,
                timeliness_score=timeliness,
                validity_score=validity,
                overall_quality=overall_quality,
                issues=issues,
                recommendations=recommendations,
                assessment_date=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Data quality assessment failed: {e}")
            raise
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate completeness score"""
        if data.empty:
            return 0.0
        
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        return completeness
    
    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate accuracy score (simplified)"""
        # Simplified accuracy calculation based on data validation rules
        accuracy = 0.85  # Default accuracy
        
        # Check for realistic values in numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            if col.lower() in ['age', 'years', 'count']:
                # Check for realistic ranges
                if data[col].min() >= 0 and data[col].max() < 200:
                    accuracy += 0.05
        
        # Check for consistent data formats
        text_cols = data.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            # Simple check for data consistency
            accuracy += 0.05
        
        return min(1.0, accuracy)
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency score"""
        consistency = 0.80  # Default consistency
        
        # Check for data type consistency
        for col in data.columns:
            non_null_values = data[col].dropna()
            if len(non_null_values) > 0:
                # Simplified consistency check
                consistency += 0.02
        
        return min(1.0, consistency)
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score"""
        timeliness = 0.90  # Default timeliness - assumes recent data
        
        # Check if data has timestamp columns
        date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            # Assess data recency
            for date_col in date_columns:
                try:
                    latest_date = pd.to_datetime(data[date_col]).max()
                    days_old = (datetime.now() - latest_date).days
                    if days_old <= 30:
                        timeliness = 0.95
                    elif days_old <= 90:
                        timeliness = 0.85
                    else:
                        timeliness = 0.75
                except:
                    continue
        
        return timeliness
    
    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate validity score"""
        validity = 0.80  # Default validity
        
        # Check for data format validation
        for col in data.select_dtypes(include=[np.number]).columns:
            # Check for reasonable numeric ranges
            if data[col].min() >= 0:
                validity += 0.02
        
        # Check for text format validation
        for col in data.select_dtypes(include=['object']).columns:
            # Simple check for reasonable string lengths
            avg_length = data[col].astype(str).str.len().mean()
            if 1 <= avg_length <= 100:  # Reasonable string length
                validity += 0.01
        
        return min(1.0, validity)
    
    def _identify_data_issues(self, data: pd.DataFrame, completeness: float, accuracy: float,
                            consistency: float, timeliness: float, validity: float) -> List[str]:
        """Identify data quality issues"""
        issues = []
        
        if completeness < 0.8:
            issues.append(f"Low data completeness: {completeness:.1%}")
        
        if accuracy < 0.8:
            issues.append("Data accuracy concerns identified")
        
        if consistency < 0.8:
            issues.append("Data consistency issues detected")
        
        if timeliness < 0.8:
            issues.append("Data may be outdated")
        
        if validity < 0.8:
            issues.append("Data validity concerns")
        
        # Check for duplicate rows
        if len(data) > 0 and len(data.duplicated()) > 0:
            issues.append(f"Duplicate rows found: {len(data.duplicated())}")
        
        # Check for outliers (simplified)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = data[(data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)]
            if len(outliers) > len(data) * 0.1:  # More than 10% outliers
                issues.append(f"Potential outliers in {col}: {len(outliers)} records")
        
        return issues
    
    def _generate_quality_recommendations(self, completeness: float, accuracy: float,
                                        consistency: float, timeliness: float, validity: float) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        if completeness < 0.8:
            recommendations.append("Implement data validation rules to reduce missing values")
            recommendations.append("Review data collection processes")
        
        if accuracy < 0.8:
            recommendations.append("Add data validation checks for input accuracy")
            recommendations.append("Implement data cleansing procedures")
        
        if consistency < 0.8:
            recommendations.append("Standardize data formats and structures")
            recommendations.append("Implement data governance policies")
        
        if timeliness < 0.8:
            recommendations.append("Increase data refresh frequency")
            recommendations.append("Implement automated data pipelines")
        
        if validity < 0.8:
            recommendations.append("Enhance data validation rules")
            recommendations.append("Implement data quality monitoring")
        
        return recommendations
    
    def _store_quality_report(self, dataset_id: str, completeness: float, accuracy: float,
                            consistency: float, timeliness: float, validity: float,
                            overall_quality: DataQuality, issues: List[str], 
                            recommendations: List[str]) -> None:
        """Store quality report in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO data_quality_reports
                (report_id, dataset_id, completeness, accuracy, consistency, timeliness, validity,
                 overall_quality, issues, recommendations, assessment_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"report_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dataset_id,
                completeness,
                accuracy,
                consistency,
                timeliness,
                validity,
                overall_quality.value,
                json.dumps(issues),
                json.dumps(recommendations),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Failed to store quality report: {e}")
    
    def get_data_summary(self, dataset_id: str) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        data = self.retrieve_data(dataset_id)
        if data is None:
            return {}
        
        summary = {
            "dataset_info": {
                "dataset_id": dataset_id,
                "record_count": len(data),
                "column_count": len(data.columns),
                "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024,
                "created_date": datetime.now().isoformat()
            },
            "column_summary": {},
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "basic_stats": {}
        }
        
        # Column summary
        for col in data.columns:
            summary["column_summary"][col] = {
                "data_type": str(data[col].dtype),
                "unique_values": data[col].nunique(),
                "missing_count": data[col].isnull().sum(),
                "missing_percentage": (data[col].isnull().sum() / len(data)) * 100
            }
        
        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["basic_stats"] = data[numeric_cols].describe().to_dict()
        
        return summary
    
    def schedule_data_ingestion(self, config: DataIngestionConfig) -> None:
        """Schedule data ingestion"""
        self.ingestion_configs[config.source_id] = config
        
        # In a real implementation, this would integrate with a job scheduler like APScheduler
        logging.info(f"Data ingestion scheduled for source: {config.source_id}")
        logging.info(f"Schedule: {config.schedule}")
    
    def validate_data_schema(self, schema: DataSchema, data: pd.DataFrame) -> Dict[str, bool]:
        """Validate data against schema"""
        validation_results = {}
        
        # Check column existence
        for col in schema.columns.keys():
            validation_results[f"column_{col}_exists"] = col in data.columns
        
        # Check data types (simplified)
        for col, expected_type in schema.columns.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                expected_type_check = self._type_compatibility_check(actual_type, expected_type)
                validation_results[f"column_{col}_type"] = expected_type_check
        
        return validation_results
    
    def _type_compatibility_check(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual data type is compatible with expected type"""
        type_compatibility = {
            "int64": ["integer", "int", "numeric"],
            "float64": ["decimal", "float", "numeric"],
            "object": ["string", "text", "varchar"],
            "datetime64[ns]": ["datetime", "timestamp", "date"]
        }
        
        expected_lower = expected_type.lower()
        actual_lower = actual_type.lower()
        
        for actual, expected_list in type_compatibility.items():
            if actual_lower in actual_type.lower():
                return any(exp.lower() in expected_lower for exp in expected_list)
        
        return True  # Default to compatible
    
    def export_data(self, dataset_id: str, format: str = "csv", 
                   output_path: Optional[str] = None) -> str:
        """Export data in specified format"""
        data = self.retrieve_data(dataset_id)
        if data is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        if output_path is None:
            output_path = f"{dataset_id}.{format}"
        
        try:
            if format.lower() == "csv":
                data.to_csv(output_path, index=False)
            elif format.lower() == "json":
                data.to_json(output_path, orient="records")
            elif format.lower() == "excel":
                data.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logging.info(f"Data exported to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Data export failed: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    data_manager = DataManager("analytics_test.db")
    
    # Register data source
    config = DataIngestionConfig(
        source_id="sample_data",
        source_type=DataSource.FILE,
        connection_string="sample_data.csv",
        schedule="0 2 * * *",  # Daily at 2 AM
        enabled=True
    )
    
    data_manager.register_data_source(config)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'revenue': np.random.uniform(100, 1000, 100),
        'date_joined': pd.date_range('2020-01-01', periods=100, freq='D')
    })
    
    # Ingest data
    dataset_id = data_manager.ingest_data("sample_data", sample_data)
    print(f"Data ingested with ID: {dataset_id}")
    
    # Get data summary
    summary = data_manager.get_data_summary(dataset_id)
    print(f"Dataset summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Get quality report
    quality_report = data_manager.quality_reports.get(dataset_id)
    if quality_report:
        print(f"Quality score: {quality_report.overall_quality.value}")
        print(f"Issues found: {len(quality_report.issues)}")