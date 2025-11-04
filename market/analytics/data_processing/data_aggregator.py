"""
Data Aggregator for Business Intelligence
Aggregates and processes data from multiple sources
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
import json
import logging
from abc import ABC, abstractmethod

class DataSource(ABC):
    """Abstract data source interface"""
    
    @abstractmethod
    def load_data(self, date_range: tuple[date, date]) -> List[Dict[str, Any]]:
        """Load data from source"""
        pass

class DatabaseDataSource(DataSource):
    """Database data source implementation"""
    
    def __init__(self, connection_string: str, config: Dict[str, Any]):
        self.connection_string = connection_string
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, date_range: tuple[date, date]) -> List[Dict[str, Any]]:
        """Load data from database"""
        # Mock implementation - would connect to actual database
        self.logger.info(f"Loading data from database for range {date_range}")
        
        # Return mock data
        return [
            {
                'customer_id': 'CUST_001',
                'company_name': 'Healthcare Corp',
                'industry': 'Healthcare',
                'acquisition_date': date(2023, 1, 15),
                'monthly_recurring_revenue': Decimal('5000'),
                'status': 'Active'
            }
        ]

class APIDataSource(DataSource):
    """API data source implementation"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.api_config = api_config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, date_range: tuple[date, date]) -> List[Dict[str, Any]]:
        """Load data from API"""
        # Mock implementation - would call actual API
        self.logger.info(f"Loading data from API for range {date_range}")
        return []

class FileDataSource(DataSource):
    """File-based data source implementation"""
    
    def __init__(self, file_path: str, format_type: str = 'json'):
        self.file_path = file_path
        self.format_type = format_type
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, date_range: tuple[date, date]) -> List[Dict[str, Any]]:
        """Load data from file"""
        self.logger.info(f"Loading data from file {self.file_path}")
        # Mock implementation - would read actual files
        return []

class DataAggregator:
    """Main data aggregation component"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize data sources
        self.data_sources = {
            'database': self._init_database_source(),
            'api': self._init_api_source(),
            'files': self._init_file_sources()
        }
        
        # Cache for performance
        self._data_cache = {}
        self._cache_expiry = {}
    
    def _init_database_source(self) -> Optional[DatabaseDataSource]:
        """Initialize database data source"""
        try:
            db_config = self.config.get('database', {})
            if db_config:
                return DatabaseDataSource(
                    connection_string=db_config.get('connection_string', ''),
                    config=db_config
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize database source: {e}")
        return None
    
    def _init_api_source(self) -> Optional[APIDataSource]:
        """Initialize API data source"""
        try:
            api_config = self.config.get('api', {})
            if api_config:
                return APIDataSource(api_config=api_config)
        except Exception as e:
            self.logger.error(f"Failed to initialize API source: {e}")
        return None
    
    def _init_file_sources(self) -> List[FileDataSource]:
        """Initialize file data sources"""
        file_sources = []
        try:
            files_config = self.config.get('files', [])
            for file_config in files_config:
                file_sources.append(FileDataSource(
                    file_path=file_config.get('path', ''),
                    format_type=file_config.get('format', 'json')
                ))
        except Exception as e:
            self.logger.error(f"Failed to initialize file sources: {e}")
        return file_sources
    
    def get_customers(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get customer data"""
        cache_key = f"customers_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Load from multiple sources
            customers = []
            
            # Database source
            if self.data_sources['database']:
                db_data = self.data_sources['database'].load_data(date_range or (date.today(), date.today()))
                customers.extend(self._transform_customer_data(db_data))
            
            # API source
            if self.data_sources['api']:
                api_data = self.data_sources['api'].load_data(date_range or (date.today(), date.today()))
                customers.extend(self._transform_customer_data(api_data))
            
            # File sources
            for file_source in self.data_sources['files']:
                file_data = file_source.load_data(date_range or (date.today(), date.today()))
                customers.extend(self._transform_customer_data(file_data))
            
            # Cache results
            self._cache_data(cache_key, customers)
            
            self.logger.info(f"Retrieved {len(customers)} customers")
            return customers
            
        except Exception as e:
            self.logger.error(f"Error getting customers: {e}")
            return []
    
    def get_customer_cohorts(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get customer cohort data"""
        cache_key = f"cohorts_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Generate cohort data from customer data
            customers = self.get_customers(date_range)
            cohorts = self._generate_cohorts(customers)
            
            self._cache_data(cache_key, cohorts)
            return cohorts
            
        except Exception as e:
            self.logger.error(f"Error getting cohorts: {e}")
            return []
    
    def get_sales_metrics(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get sales metrics data"""
        cache_key = f"sales_metrics_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock sales metrics data
            sales_metrics = self._generate_mock_sales_metrics(date_range)
            
            self._cache_data(cache_key, sales_metrics)
            return sales_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting sales metrics: {e}")
            return []
    
    def get_pipeline_data(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get pipeline data"""
        cache_key = f"pipeline_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock pipeline data
            pipeline = self._generate_mock_pipeline_data(date_range)
            
            self._cache_data(cache_key, pipeline)
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline data: {e}")
            return []
    
    def get_marketing_metrics(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get marketing metrics data"""
        cache_key = f"marketing_metrics_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock marketing metrics
            marketing_metrics = self._generate_mock_marketing_metrics(date_range)
            
            self._cache_data(cache_key, marketing_metrics)
            return marketing_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting marketing metrics: {e}")
            return []
    
    def get_campaign_data(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get campaign data"""
        cache_key = f"campaigns_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock campaign data
            campaigns = self._generate_mock_campaign_data(date_range)
            
            self._cache_data(cache_key, campaigns)
            return campaigns
            
        except Exception as e:
            self.logger.error(f"Error getting campaign data: {e}")
            return []
    
    def get_cac_analysis(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get CAC analysis data"""
        cache_key = f"cac_analysis_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock CAC analysis
            cac_data = self._generate_mock_cac_analysis(date_range)
            
            self._cache_data(cache_key, cac_data)
            return cac_data
            
        except Exception as e:
            self.logger.error(f"Error getting CAC analysis: {e}")
            return []
    
    def get_market_share(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get market share data"""
        cache_key = f"market_share_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock market share data
            market_share = self._generate_mock_market_share(date_range)
            
            self._cache_data(cache_key, market_share)
            return market_share
            
        except Exception as e:
            self.logger.error(f"Error getting market share data: {e}")
            return []
    
    def get_competitive_data(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get competitive analysis data"""
        cache_key = f"competitive_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock competitive data
            competitive = self._generate_mock_competitive_data(date_range)
            
            self._cache_data(cache_key, competitive)
            return competitive
            
        except Exception as e:
            self.logger.error(f"Error getting competitive data: {e}")
            return []
    
    def get_benchmarking_data(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get benchmarking data"""
        cache_key = f"benchmarking_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock benchmarking data
            benchmarking = self._generate_mock_benchmarking_data(date_range)
            
            self._cache_data(cache_key, benchmarking)
            return benchmarking
            
        except Exception as e:
            self.logger.error(f"Error getting benchmarking data: {e}")
            return []
    
    def get_revenue_history(self, date_range: Optional[tuple[date, date]] = None) -> List[Dict[str, Any]]:
        """Get revenue history data"""
        cache_key = f"revenue_history_{date_range}"
        
        if self._is_cache_valid(cache_key):
            return self._data_cache[cache_key]
        
        try:
            # Mock revenue history
            revenue_history = self._generate_mock_revenue_history(date_range)
            
            self._cache_data(cache_key, revenue_history)
            return revenue_history
            
        except Exception as e:
            self.logger.error(f"Error getting revenue history: {e}")
            return []
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._data_cache:
            return False
        
        # Check expiry (5 minutes cache)
        expiry_time = self._cache_expiry.get(cache_key)
        if expiry_time and datetime.now() > expiry_time:
            return False
        
        return True
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with expiry"""
        self._data_cache[cache_key] = data
        self._cache_expiry[cache_key] = datetime.now().replace(minute=datetime.now().minute + 5)
    
    def _transform_customer_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform raw customer data"""
        # Mock transformation
        return raw_data
    
    def _generate_cohorts(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate cohort analysis from customer data"""
        cohorts = {}
        
        for customer in customers:
            acquisition_date = customer.get('acquisition_date')
            if isinstance(acquisition_date, str):
                acquisition_date = datetime.strptime(acquisition_date, '%Y-%m-%d').date()
            
            cohort_month = acquisition_date.strftime('%Y-%m')
            
            if cohort_month not in cohorts:
                cohorts[cohort_month] = {
                    'cohort_month': cohort_month,
                    'customers': [],
                    'total_revenue': Decimal('0'),
                    'retention_rates': {}
                }
            
            cohorts[cohort_month]['customers'].append(customer)
            cohorts[cohort_month]['total_revenue'] += Decimal(str(customer.get('monthly_recurring_revenue', '0')))
        
        return list(cohorts.values())
    
    def _generate_mock_sales_metrics(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock sales metrics"""
        return [
            {
                'period_start': date_range[0] if date_range else date(2024, 1, 1),
                'period_end': date_range[1] if date_range else date(2024, 12, 31),
                'total_leads': 1000,
                'qualified_leads': 250,
                'proposals_sent': 100,
                'deals_won': 30,
                'total_pipeline_value': Decimal('750000'),
                'closed_won_value': Decimal('300000'),
                'revenue': Decimal('300000')
            }
        ]
    
    def _generate_mock_pipeline_data(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock pipeline data"""
        return [
            {
                'deal_id': 'DEAL_001',
                'customer_name': 'Tech Corp',
                'deal_value': Decimal('50000'),
                'stage': 'qualified',
                'probability': 0.25,
                'expected_close_date': date(2025, 3, 15),
                'sales_rep': 'John Smith',
                'days_in_stage': 15
            },
            {
                'deal_id': 'DEAL_002',
                'customer_name': 'Health Systems',
                'deal_value': Decimal('75000'),
                'stage': 'proposal',
                'probability': 0.50,
                'expected_close_date': date(2025, 4, 30),
                'sales_rep': 'Sarah Johnson',
                'days_in_stage': 30
            }
        ]
    
    def _generate_mock_marketing_metrics(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock marketing metrics"""
        return [
            {
                'period_start': date_range[0] if date_range else date(2024, 1, 1),
                'period_end': date_range[1] if date_range else date(2024, 12, 31),
                'website_visitors': 10000,
                'leads_generated': 500,
                'total_marketing_spend': Decimal('50000'),
                'cost_per_lead': Decimal('100')
            }
        ]
    
    def _generate_mock_campaign_data(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock campaign data"""
        return [
            {
                'campaign_id': 'CAMP_001',
                'campaign_name': 'Q1 Product Launch',
                'campaign_type': 'email',
                'budget': Decimal('10000'),
                'actual_spend': Decimal('8500'),
                'impressions': 50000,
                'clicks': 2500,
                'conversions': 125,
                'leads_generated': 100
            }
        ]
    
    def _generate_mock_cac_analysis(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock CAC analysis"""
        return [
            {
                'period_start': date_range[0] if date_range else date(2024, 1, 1),
                'period_end': date_range[1] if date_range else date(2024, 12, 31),
                'total_customer_acquisitions': 50,
                'total_acquisition_cost': Decimal('25000'),
                'average_cac': Decimal('500'),
                'overall_ltv_cac_ratio': 3.2
            }
        ]
    
    def _generate_mock_market_share(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock market share data"""
        return [
            {
                'period_start': date_range[0] if date_range else date(2024, 1, 1),
                'period_end': date_range[1] if date_range else date(2024, 12, 31),
                'market_name': 'Healthcare AI Software',
                'our_market_share': 0.15,
                'market_growth_rate': 0.12
            }
        ]
    
    def _generate_mock_competitive_data(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock competitive data"""
        return [
            {
                'competitor_id': 'COMP_001',
                'competitor_name': 'Competitor Corp',
                'competitor_type': 'direct',
                'market_share': 0.25,
                'pricing_competitiveness': 75
            }
        ]
    
    def _generate_mock_benchmarking_data(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock benchmarking data"""
        return [
            {
                'metric_name': 'Customer Satisfaction',
                'metric_value': 8.5,
                'metric_unit': 'score',
                'industry_average': 7.8,
                'best_in_class': 9.2,
                'percentile_rank': 85
            }
        ]
    
    def _generate_mock_revenue_history(self, date_range: Optional[tuple[date, date]]) -> List[Dict[str, Any]]:
        """Generate mock revenue history"""
        revenue_history = []
        start_date = date_range[0] if date_range else date(2023, 1, 1)
        end_date = date_range[1] if date_range else date(2024, 12, 31)
        
        current_date = start_date
        base_revenue = Decimal('500000')
        
        while current_date <= end_date:
            # Add some growth and seasonality
            month_offset = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
            seasonal_factor = 1 + 0.1 * (1 if current_date.month in [11, 12] else 0)  # Holiday boost
            growth_factor = 1 + (month_offset * 0.02)  # 2% monthly growth
            
            monthly_revenue = base_revenue * Decimal(str(seasonal_factor * growth_factor))
            
            revenue_history.append({
                'date': current_date,
                'revenue': monthly_revenue,
                'new_customers': max(1, int(monthly_revenue / 10000)),
                'churned_customers': max(0, int(monthly_revenue / 20000))
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return revenue_history
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._data_cache.clear()
        self._cache_expiry.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_entries': len(self._data_cache),
            'cache_size_mb': len(str(self._data_cache)) / (1024 * 1024),
            'sources_configured': len([s for s in self.data_sources.values() if s])
        }