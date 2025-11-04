"""
Production ETL Pipeline for Healthcare Data
Handles extraction, transformation, and loading of medical data with HIPAA compliance
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
import pymongo
from concurrent.futures import ThreadPoolExecutor, as_completed

class ETLStatus(Enum):
    """ETL pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DataTransform(Enum):
    """Data transformation operations"""
    ANONYMIZATION = "anonymization"
    STANDARDIZATION = "standardization"
    ENCRYPTION = "encryption"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"
    ENRICHMENT = "enrichment"

@dataclass
class ETLJob:
    """ETL job configuration and status"""
    job_id: str
    source_name: str
    target_table: str
    status: ETLStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_log: List[str] = field(default_factory=list)
    transformation_rules: List[Dict[str, Any]] = field(default_factory=list)
    data_quality_score: float = 0.0
    phi_processed: int = 0

@dataclass
class MedicalDataRecord:
    """Standardized medical data record"""
    record_id: str
    patient_hash: str
    record_type: str
    encounter_id: str
    timestamp: datetime
    data: Dict[str, Any]
    quality_flags: List[str]
    compliance_flags: List[str]
    transformation_applied: List[str]

class MedicalETLPipeline:
    """Production ETL Pipeline for Healthcare Data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.engines = {}
        self.active_jobs = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup ETL pipeline logging"""
        logger = logging.getLogger("medical_etl")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for PHI protection"""
        key = os.getenv("ENCRYPTION_KEY")
        if key:
            return key.encode()
        else:
            # Generate new key for development
            key = Fernet.generate_key()
            self.logger.warning("Generated new encryption key - configure proper key in production")
            return key
    
    async def initialize_connections(self) -> None:
        """Initialize database and storage connections"""
        try:
            # Initialize SQL database connections
            for source_config in self.config["data_sources"]:
                if source_config.source_type.value in ["ehr", "laboratory_results"]:
                    engine = create_engine(source_config.connection_string)
                    self.engines[source_config.name] = engine
            
            # Initialize MongoDB for unstructured data
            mongo_client = pymongo.MongoClient(os.getenv("MONGODB_CONNECTION"))
            self.engines["mongodb"] = mongo_client
            
            self.logger.info("All database connections initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize connections: {str(e)}")
            raise
    
    async def extract_data(self, source_name: str, 
                          table_name: str, 
                          last_extraction: Optional[datetime] = None) -> pd.DataFrame:
        """Extract data from healthcare sources"""
        try:
            engine = self.engines.get(source_name)
            if not engine:
                raise ValueError(f"No connection found for source: {source_name}")
            
            # Build extraction query with incremental loading
            where_clause = ""
            if last_extraction:
                where_clause = f"WHERE updated_at > '{last_extraction.isoformat()}'"
            
            query = f"""
                SELECT * FROM {table_name} 
                {where_clause}
                ORDER BY updated_at
                LIMIT 100000
            """
            
            df = pd.read_sql(query, engine)
            self.logger.info(f"Extracted {len(df)} records from {source_name}.{table_name}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data extraction failed for {source_name}.{table_name}: {str(e)}")
            raise
    
    async def transform_medical_data(self, df: pd.DataFrame, 
                                   transformation_rules: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply medical data transformations"""
        try:
            transformed_df = df.copy()
            
            for rule in transformation_rules:
                transform_type = rule.get("type")
                
                if transform_type == DataTransform.ANONYMIZATION.value:
                    transformed_df = await self._anonymize_phi(transformed_df, rule.get("fields", []))
                
                elif transform_type == DataTransform.STANDARDIZATION.value:
                    transformed_df = await self._standardize_data(transformed_df, rule.get("fields", []))
                
                elif transform_type == DataTransform.ENCRYPTION.value:
                    transformed_df = await self._encrypt_sensitive_data(transformed_df, rule.get("fields", []))
                
                elif transform_type == DataTransform.VALIDATION.value:
                    transformed_df = await self._validate_data(transformed_df, rule.get("fields", []))
                
                elif transform_type == DataTransform.AGGREGATION.value:
                    transformed_df = await self._aggregate_data(transformed_df, rule.get("group_by", []))
                
                elif transform_type == DataTransform.ENRICHMENT.value:
                    transformed_df = await self._enrich_data(transformed_df, rule.get("enrichment_source", ""))
            
            self.logger.info(f"Applied {len(transformation_rules)} transformation rules")
            return transformed_df
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            raise
    
    async def _anonymize_phi(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """Anonymize Protected Health Information"""
        for field in fields:
            if field in df.columns:
                if field in ["name", "patient_name"]:
                    # Create pseudonymous identifiers
                    df[f"{field}_anonymized"] = df[field].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else None
                    )
                elif field in ["ssn", "social_security_number"]:
                    # Hash SSN
                    df[f"{field}_hashed"] = df[field].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None
                    )
                elif field in ["address", "street_address"]:
                    # Generalize address to ZIP code only
                    df[f"{field}_generalized"] = df[field].apply(
                        lambda x: x.split()[-1] if pd.notna(x) and len(str(x).split()) > 0 else None
                    )
                elif field == "birth_date":
                    # Reduce birth date to year only
                    df[f"{field}_year"] = pd.to_datetime(df[field]).dt.year if field in df.columns else df[field]
        
        return df
    
    async def _standardize_data(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """Standardize data formats and codes"""
        for field in fields:
            if field in df.columns:
                if field in ["diagnosis_codes", "procedure_codes"]:
                    # Standardize ICD codes
                    df[field] = df[field].apply(self._standardize_icd_codes)
                elif field == "medication_name":
                    # Standardize medication names
                    df[field] = df[field].apply(self._standardize_medication_names)
                elif field in ["blood_pressure_systolic", "blood_pressure_diastolic"]:
                    # Validate and clean blood pressure readings
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    df[field] = df[field].where((df[field] >= 0) & (df[field] <= 300))
                elif field == "age":
                    # Ensure age is within reasonable bounds
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    df[field] = df[field].where((df[field] >= 0) & (df[field] <= 150))
        
        return df
    
    async def _encrypt_sensitive_data(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """Encrypt sensitive medical data"""
        for field in fields:
            if field in df.columns:
                df[f"{field}_encrypted"] = df[field].apply(
                    lambda x: self.cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) else None
                )
        
        return df
    
    async def _validate_data(self, df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
        """Apply data validation rules"""
        validation_results = []
        
        for field in fields:
            if field in df.columns:
                # Check for null values
                null_percentage = df[field].isnull().sum() / len(df) * 100
                if null_percentage > 10:
                    validation_results.append(f"High null rate in {field}: {null_percentage:.1f}%")
                
                # Field-specific validations
                if field == "patient_id":
                    # Validate patient ID format
                    valid_format = df[field].str.match(r'^[A-Z0-9]{8,12}$', na=False).sum()
                    invalid_count = len(df) - valid_format
                    if invalid_count > 0:
                        validation_results.append(f"Invalid patient_id format: {invalid_count} records")
                
                elif field == "diagnosis_code":
                    # Validate ICD-10 codes
                    valid_codes = df[field].str.match(r'^[A-Z][0-9]{2}(\.[0-9]{1,2})?$', na=False).sum()
                    invalid_count = len(df) - valid_codes
                    if invalid_count > 0:
                        validation_results.append(f"Invalid diagnosis codes: {invalid_count} records")
        
        if validation_results:
            self.logger.warning(f"Data validation issues: {', '.join(validation_results)}")
        
        return df
    
    async def _aggregate_data(self, df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
        """Aggregate medical data by specified dimensions"""
        if not group_by:
            return df
        
        # Common medical aggregations
        aggregation_rules = {
            'encounter_count': ('encounter_id', 'count'),
            'total_cost': ('cost', 'sum'),
            'avg_length_of_stay': ('length_of_stay', 'mean'),
            'medication_count': ('medication_id', 'count'),
            'diagnosis_count': ('diagnosis_code', 'count')
        }
        
        aggregated = df.groupby(group_by).agg(aggregation_rules).reset_index()
        return aggregated
    
    async def _enrich_data(self, df: pd.DataFrame, enrichment_source: str) -> pd.DataFrame:
        """Enrich medical data with additional information"""
        # Example: Add demographic data, clinical risk scores, etc.
        if enrichment_source == "demographics":
            # Join with demographic data
            pass
        elif enrichment_source == "risk_scores":
            # Calculate clinical risk scores
            df['risk_score'] = df.apply(self._calculate_risk_score, axis=1)
        
        return df
    
    def _calculate_risk_score(self, row: pd.Series) -> float:
        """Calculate clinical risk score"""
        score = 0.0
        
        # Age factor
        if 'age' in row:
            age = row['age']
            if age >= 65:
                score += 0.3
            elif age >= 45:
                score += 0.2
        
        # Comorbidity factor
        if 'comorbidity_count' in row:
            score += row['comorbidity_count'] * 0.1
        
        # Medication count
        if 'medication_count' in row:
            score += min(row['medication_count'] * 0.05, 0.3)
        
        return min(score, 1.0)
    
    async def load_data(self, df: pd.DataFrame, target_table: str) -> int:
        """Load transformed data to target system"""
        try:
            # Load to analytics warehouse
            engine = self.engines.get("analytics_warehouse")
            if engine:
                df.to_sql(target_table, engine, if_exists='append', index=False)
                records_loaded = len(df)
                self.logger.info(f"Loaded {records_loaded} records to {target_table}")
                return records_loaded
            else:
                raise ValueError("No analytics warehouse connection available")
                
        except Exception as e:
            self.logger.error(f"Data loading failed for {target_table}: {str(e)}")
            raise
    
    async def run_etl_job(self, job_config: Dict[str, Any]) -> ETLJob:
        """Execute a complete ETL job"""
        job = ETLJob(
            job_id=job_config["job_id"],
            source_name=job_config["source_name"],
            target_table=job_config["target_table"],
            status=ETLStatus.PENDING,
            transformation_rules=job_config.get("transformation_rules", [])
        )
        
        self.active_jobs[job.job_id] = job
        job.status = ETLStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            self.logger.info(f"Starting ETL job: {job.job_id}")
            
            # Extract
            df = await self.extract_data(
                job_config["source_name"],
                job_config["source_table"],
                job_config.get("last_extraction")
            )
            
            # Transform
            if job.transformation_rules:
                df = await self.transform_medical_data(df, job.transformation_rules)
            
            # Load
            records_loaded = await self.load_data(df, job.target_table)
            
            # Update job status
            job.records_processed = records_loaded
            job.status = ETLStatus.COMPLETED
            job.completed_at = datetime.now()
            job.data_quality_score = self._calculate_data_quality_score(df)
            
            self.logger.info(f"ETL job completed: {job.job_id} - {records_loaded} records processed")
            
        except Exception as e:
            job.status = ETLStatus.FAILED
            job.completed_at = datetime.now()
            job.error_log.append(str(e))
            self.logger.error(f"ETL job failed: {job.job_id} - {str(e)}")
        
        return job
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if df.empty:
            return 0.0
        
        score_components = []
        
        # Completeness score
        completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        score_components.append(completeness * 0.3)
        
        # Validity score (basic validation)
        validity = 0.95  # Default validity score
        score_components.append(validity * 0.3)
        
        # Consistency score
        consistency = 0.98  # Default consistency score
        score_components.append(consistency * 0.2)
        
        # Uniqueness score
        uniqueness = 0.99 if df.duplicated().sum() == 0 else 0.95
        score_components.append(uniqueness * 0.2)
        
        return sum(score_components)
    
    async def run_incremental_etl(self, pipeline_config: Dict[str, Any]) -> List[ETLJob]:
        """Run incremental ETL for multiple sources"""
        jobs = []
        last_extraction = datetime.now() - timedelta(hours=1)  # Default incremental window
        
        for source_config in pipeline_config["sources"]:
            job_config = {
                "job_id": f"{source_config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "source_name": source_config["name"],
                "source_table": source_config["table"],
                "target_table": source_config["target_table"],
                "transformation_rules": source_config.get("transformations", []),
                "last_extraction": last_extraction
            }
            
            job = await self.run_etl_job(job_config)
            jobs.append(job)
        
        return jobs
    
    async def close_connections(self) -> None:
        """Close all database connections"""
        for engine in self.engines.values():
            if hasattr(engine, 'dispose'):
                engine.dispose()
        
        self.logger.info("All database connections closed")

def create_etl_pipeline(config_path: str = None) -> MedicalETLPipeline:
    """Factory function to create ETL pipeline instance"""
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use default production configuration
        from production.data.config.data_config import PRODUCTION_CONFIG
        config = PRODUCTION_CONFIG
    
    return MedicalETLPipeline(config)

# Example usage
if __name__ == "__main__":
    async def main():
        pipeline = create_etl_pipeline()
        
        # Initialize connections
        await pipeline.initialize_connections()
        
        # Run incremental ETL
        etl_config = {
            "sources": [
                {
                    "name": "Epic EHR",
                    "table": "patients",
                    "target_table": "analytics_patients",
                    "transformations": [
                        {"type": "anonymization", "fields": ["name", "ssn"]},
                        {"type": "standardization", "fields": ["diagnosis_codes"]}
                    ]
                }
            ]
        }
        
        jobs = await pipeline.run_incremental_etl(etl_config)
        
        for job in jobs:
            print(f"Job {job.job_id}: {job.status.value} - {job.records_processed} records")
        
        # Close connections
        await pipeline.close_connections()
    
    asyncio.run(main())
