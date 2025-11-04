# Analytics Platform Test Suite
# Comprehensive testing for all analytics modules

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import os

# Import all analytics modules
from core.analytics_engine import AdvancedAnalyticsEngine, AnalyticsType
from predictive.forecast_engine import PredictiveAnalytics
from customer.behavior_analytics import CustomerAnalytics, SegmentType
from market.intelligence_engine import MarketIntelligenceEngine
from clinical.outcome_analytics import ClinicalAnalytics
from operational.efficiency_analytics import OperationalAnalytics
from executive.intelligence_system import ExecutiveIntelligence
from data.data_manager import DataManager, DataIngestionConfig, DataSource
from config.configuration import ConfigManager, AnalyticsConfig
from core.orchestrator import AnalyticsOrchestrator

class TestAnalyticsEngine:
    """Test the core analytics engine"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'revenue': [100000, 120000, 115000, 140000, 135000],
            'customers': [500, 600, 580, 720, 690],
            'satisfaction': [4.1, 4.3, 4.0, 4.5, 4.2],
            'date': pd.date_range('2023-01-01', periods=5, freq='M')
        })
    
    @pytest.fixture
    def analytics_engine(self):
        """Create analytics engine instance"""
        return AdvancedAnalyticsEngine()
    
    def test_engine_initialization(self, analytics_engine):
        """Test analytics engine initialization"""
        assert analytics_engine is not None
        assert analytics_engine.config is not None
        assert len(analytics_engine.models) == 0
    
    def test_data_quality_assessment(self, analytics_engine, sample_data):
        """Test data quality assessment"""
        quality_report = analytics_engine._assess_data_quality(sample_data)
        
        assert 'overall_quality_score' in quality_report
        assert 'metrics' in quality_report
        assert 'issues' in quality_report
        assert 'recommendations' in quality_report
        
        # Check quality metrics exist
        metrics = quality_report['metrics']
        assert 'completeness' in metrics
        assert 'consistency' in metrics
        assert 'accuracy' in metrics
        
        assert 0 <= quality_report['overall_quality_score'] <= 1
    
    def test_predictive_insights_generation(self, analytics_engine, sample_data):
        """Test predictive insights generation"""
        insights = analytics_engine._generate_predictive_insights(sample_data)
        
        assert len(insights) > 0
        for insight in insights:
            assert hasattr(insight, 'insight_id')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'confidence_score')
            assert 0 <= insight.confidence_score <= 1
    
    def test_full_analytics_processing(self, analytics_engine, sample_data):
        """Test complete analytics processing"""
        result = analytics_engine.process_data(sample_data, AnalyticsType.PREDICTIVE)
        
        assert 'quality_report' in result
        assert 'insights' in result
        assert 'analytics_score' in result
        assert 'recommendations' in result
        assert 'timestamp' in result
        
        assert len(result['insights']) > 0
        assert result['analytics_score'] >= 0

class TestPredictiveAnalytics:
    """Test predictive analytics module"""
    
    @pytest.fixture
    def predictive_analytics(self):
        """Create predictive analytics instance"""
        return PredictiveAnalytics()
    
    @pytest.fixture
    def time_series_data(self):
        """Create time series data for forecasting"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100) * 100 + 1000)
        
        return pd.DataFrame({
            'date': dates,
            'revenue': values,
            'customers': np.random.randint(100, 1000, 100)
        })
    
    def test_forecast_model_creation(self, predictive_analytics, time_series_data):
        """Test forecast model creation"""
        model_id = predictive_analytics.create_forecast_model(
            time_series_data, 'revenue', model_type='random_forest'
        )
        
        assert model_id is not None
        assert model_id in predictive_analytics.models
        assert model_id in predictive_analytics.scalers
    
    def test_forecast_generation(self, predictive_analytics, time_series_data):
        """Test forecast generation"""
        model_id = predictive_analytics.create_forecast_model(
            time_series_data, 'revenue'
        )
        
        forecasts = predictive_analytics.generate_forecast(model_id, forecast_periods=7)
        
        assert len(forecasts) == 7
        for forecast in forecasts:
            assert forecast.predicted_value > 0
            assert forecast.confidence_interval[0] <= forecast.predicted_value <= forecast.confidence_interval[1]
    
    def test_trend_analysis(self, predictive_analytics, time_series_data):
        """Test trend analysis"""
        trends = predictive_analytics.analyze_trends(time_series_data, 'revenue')
        
        assert len(trends) > 0
        for trend in trends:
            assert hasattr(trend, 'trend_id')
            assert hasattr(trend, 'trend_type')
            assert hasattr(trend, 'strength')
            assert 0 <= trend.strength <= 1

class TestCustomerAnalytics:
    """Test customer analytics module"""
    
    @pytest.fixture
    def customer_analytics(self):
        """Create customer analytics instance"""
        return CustomerAnalytics()
    
    @pytest.fixture
    def customer_data(self):
        """Create customer data"""
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100)
        })
    
    @pytest.fixture
    def transaction_data(self):
        """Create transaction data"""
        return pd.DataFrame({
            'customer_id': np.random.choice(range(1, 101), 500),
            'date': pd.date_range('2023-01-01', periods=500, freq='D'),
            'amount': np.random.uniform(10, 500, 500)
        })
    
    def test_rfm_segmentation(self, customer_analytics, customer_data):
        """Test RFM segmentation"""
        segments = customer_analytics.perform_customer_segmentation(
            customer_data, SegmentType.RFM, n_segments=5
        )
        
        assert len(segments) == 5
        for segment in segments:
            assert hasattr(segment, 'segment_id')
            assert hasattr(segment, 'segment_name')
            assert segment.customer_count > 0
    
    def test_customer_behavior_analysis(self, customer_analytics, customer_data, transaction_data):
        """Test customer behavior analysis"""
        insights = customer_analytics.analyze_customer_behavior(
            customer_data, transaction_data
        )
        
        assert len(insights) > 0
        for insight in insights:
            assert hasattr(insight, 'insight_id')
            assert hasattr(insight, 'title')
            assert 0 <= insight.impact_score <= 1
    
    def test_clv_calculation(self, customer_analytics, customer_data, transaction_data):
        """Test Customer Lifetime Value calculation"""
        clv_data = customer_analytics.calculate_customer_lifetime_value(
            customer_data, transaction_data
        )
        
        assert isinstance(clv_data, pd.DataFrame)
        assert 'predicted_clv' in clv_data.columns
        assert len(clv_data) > 0
        assert (clv_data['predicted_clv'] >= 0).all()
    
    def test_churn_prediction(self, customer_analytics, customer_data, transaction_data):
        """Test churn prediction"""
        churn_predictions = customer_analytics.predict_churn_probability(
            customer_data, transaction_data
        )
        
        assert isinstance(churn_predictions, pd.DataFrame)
        assert 'churn_probability' in churn_predictions.columns
        assert (churn_predictions['churn_probability'] >= 0).all()
        assert (churn_predictions['churn_probability'] <= 1).all()

class TestMarketIntelligence:
    """Test market intelligence module"""
    
    @pytest.fixture
    def market_intelligence(self):
        """Create market intelligence instance"""
        return MarketIntelligenceEngine()
    
    @pytest.fixture
    def market_data(self):
        """Create market data"""
        return pd.DataFrame({
            'market_name': ['Global Software Market'] * 50,
            'revenue': np.random.uniform(1000000, 10000000, 50),
            'growth_rate': np.random.uniform(0.02, 0.20, 50)
        })
    
    @pytest.fixture
    def competitor_data(self):
        """Create competitor data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['TechCorp', 'InnovateCorp', 'GlobalSoft', 'DataCorp'],
            'revenue': [50000000, 35000000, 25000000, 15000000],
            'growth_rate': [0.15, 0.08, 0.12, 0.05],
            'market_share': [0.35, 0.25, 0.20, 0.15]
        })
    
    def test_market_analysis(self, market_intelligence, market_data):
        """Test market analysis"""
        market_analysis = market_intelligence.conduct_market_analysis(market_data)
        
        assert hasattr(market_analysis, 'market_id')
        assert hasattr(market_analysis, 'total_market_size')
        assert hasattr(market_analysis, 'growth_rate')
        assert hasattr(market_analysis, 'trend')
        assert len(market_analysis.market_opportunities) > 0
        assert len(market_analysis.threats) > 0
    
    def test_competitive_analysis(self, market_intelligence, competitor_data):
        """Test competitive analysis"""
        competitive_analysis = market_intelligence.analyze_competition(competitor_data)
        
        assert len(competitive_analysis) > 0
        for analysis in competitive_analysis:
            assert hasattr(analysis, 'competitor_id')
            assert hasattr(analysis, 'market_share')
            assert hasattr(analysis, 'position')
            assert 0 <= analysis.market_share <= 1
    
    def test_market_trend_monitoring(self, market_intelligence):
        """Test market trend monitoring"""
        trends = market_intelligence.monitor_market_trends([
            "Industry Reports", "News Sources"
        ])
        
        assert len(trends) > 0
        for trend in trends:
            assert hasattr(trend, 'insight_id')
            assert hasattr(trend, 'title')
            assert hasattr(trend, 'confidence_score')
            assert 0 <= trend.confidence_score <= 1

class TestClinicalAnalytics:
    """Test clinical analytics module"""
    
    @pytest.fixture
    def clinical_analytics(self):
        """Create clinical analytics instance"""
        return ClinicalAnalytics()
    
    @pytest.fixture
    def patient_data(self):
        """Create patient data"""
        return pd.DataFrame({
            'patient_id': range(1, 101),
            'age': np.random.randint(18, 90, 100),
            'comorbidity_count': np.random.randint(0, 5, 100)
        })
    
    @pytest.fixture
    def clinical_data(self):
        """Create clinical data"""
        return pd.DataFrame({
            'patient_id': range(1, 101),
            'readmission_rate': np.random.uniform(0.05, 0.25, 100),
            'mortality_rate': np.random.uniform(0.005, 0.03, 100),
            'length_of_stay': np.random.uniform(2, 8, 100),
            'patient_satisfaction': np.random.uniform(3.0, 5.0, 100)
        })
    
    @pytest.fixture
    def admission_data(self):
        """Create admission data"""
        return pd.DataFrame({
            'patient_id': range(1, 101),
            'admission_date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'length_of_stay': np.random.uniform(2, 10, 100)
        })
    
    def test_clinical_outcome_analysis(self, clinical_analytics, patient_data, clinical_data):
        """Test clinical outcome analysis"""
        outcome_measures = ['readmission_rate', 'length_of_stay', 'patient_satisfaction']
        results = clinical_analytics.analyze_clinical_outcomes(
            patient_data, clinical_data, outcome_measures
        )
        
        assert 'outcome_analysis' in results
        assert 'quality_metrics' in results
        assert len(results['outcome_analysis']) == len(outcome_measures)
    
    def test_readmission_risk_prediction(self, clinical_analytics, patient_data, admission_data):
        """Test readmission risk prediction"""
        risk_predictions = clinical_analytics.predict_readmission_risk(
            patient_data, admission_data
        )
        
        assert isinstance(risk_predictions, pd.DataFrame)
        assert 'readmission_risk' in risk_predictions.columns
        assert 'risk_level' in risk_predictions.columns
        assert (risk_predictions['readmission_risk'] >= 0).all()
        assert (risk_predictions['readmission_risk'] <= 1).all()
    
    def test_clinical_insights_generation(self, clinical_analytics, clinical_data):
        """Test clinical insights generation"""
        insights = clinical_analytics.generate_clinical_insights(
            clinical_data, clinical_data
        )
        
        assert len(insights) > 0
        for insight in insights:
            assert hasattr(insight, 'insight_id')
            assert hasattr(insight, 'title')
            assert hasattr(insight, 'evidence_strength')
            assert 0 <= insight.evidence_strength <= 1

class TestOperationalAnalytics:
    """Test operational analytics module"""
    
    @pytest.fixture
    def operational_analytics(self):
        """Create operational analytics instance"""
        return OperationalAnalytics()
    
    @pytest.fixture
    def kpi_definitions(self):
        """Create KPI definitions"""
        return [
            {
                'kpi_id': 'efficiency',
                'kpi_name': 'Operational Efficiency',
                'category': 'operational',
                'current_value': 78,
                'target_value': 85,
                'benchmark_value': 80,
                'unit': '%',
                'historical_values': [75, 76, 77, 78]
            }
        ]
    
    @pytest.fixture
    def operational_data(self):
        """Create operational data"""
        return pd.DataFrame({
            'process_id': range(1, 101),
            'efficiency': np.random.uniform(60, 95, 100),
            'cost': np.random.uniform(1000, 10000, 100),
            'output': np.random.uniform(80, 120, 100),
            'input': np.random.uniform(100, 100, 100)
        })
    
    def test_kpi_definition(self, operational_analytics, kpi_definitions):
        """Test KPI definition"""
        kpis = operational_analytics.define_kpis(kpi_definitions)
        
        assert len(kpis) > 0
        for kpi in kpis:
            assert hasattr(kpi, 'kpi_id')
            assert hasattr(kpi, 'performance_level')
            assert kpi.current_value >= 0
    
    def test_operational_efficiency_analysis(self, operational_analytics, operational_data, kpi_definitions):
        """Test operational efficiency analysis"""
        kpis = operational_analytics.define_kpis(kpi_definitions)
        analysis = operational_analytics.analyze_operational_efficiency(
            operational_data, kpis
        )
        
        assert 'overall_efficiency_score' in analysis
        assert 'category_analysis' in analysis
        assert 'improvement_opportunities' in analysis
        assert 0 <= analysis['overall_efficiency_score'] <= 1
    
    def test_performance_health_monitoring(self, operational_analytics, kpi_definitions):
        """Test performance health monitoring"""
        kpis = operational_analytics.define_kpis(kpi_definitions)
        alert_thresholds = {'efficiency': 0.15}
        
        health_status = operational_analytics.monitor_performance_health(
            kpis, alert_thresholds
        )
        
        assert 'overall_health_score' in health_status
        assert 'health_status' in health_status
        assert 'recommendations' in health_status

class TestExecutiveIntelligence:
    """Test executive intelligence module"""
    
    @pytest.fixture
    def executive_intelligence(self):
        """Create executive intelligence instance"""
        return ExecutiveIntelligence()
    
    @pytest.fixture
    def business_data(self):
        """Create business data"""
        return pd.DataFrame({
            'revenue': [100000000, 110000000, 115000000, 125000000],
            'profit': [20000000, 22000000, 25000000, 28000000],
            'market_share': [0.32, 0.33, 0.34, 0.35]
        })
    
    @pytest.fixture
    def strategic_objectives(self):
        """Create strategic objectives"""
        return {
            'revenue_growth_target': 0.20,
            'market_share_target': 0.40,
            'profitability_target': 0.25
        }
    
    def test_executive_dashboard_creation(self, executive_intelligence, business_data, strategic_objectives):
        """Test executive dashboard creation"""
        dashboard = executive_intelligence.define_executive_dashboard(
            business_data, strategic_objectives
        )
        
        assert 'executive_summary' in dashboard
        assert 'strategic_kpis' in dashboard
        assert 'strategic_priorities' in dashboard
        assert 'financial_health' in dashboard
        assert len(dashboard['strategic_kpis']) > 0
    
    def test_strategic_analysis(self, executive_intelligence, business_data):
        """Test strategic analysis"""
        market_data = pd.DataFrame({'market_size': [1000000000], 'growth_rate': [0.15]})
        competitive_data = pd.DataFrame({'competitor_share': [0.25]})
        
        analysis = executive_intelligence.conduct_strategic_analysis(
            business_data, market_data, competitive_data
        )
        
        assert 'swot_analysis' in analysis
        assert 'porter_analysis' in analysis
        assert 'strategic_positioning' in analysis
        assert len(analysis['swot_analysis']['strengths']) > 0
    
    def test_strategic_recommendations(self, executive_intelligence, business_data, strategic_objectives):
        """Test strategic recommendations"""
        market_data = pd.DataFrame({'market_size': [1000000000], 'growth_rate': [0.15]})
        competitive_data = pd.DataFrame({'competitor_share': [0.25]})
        
        strategic_analysis = executive_intelligence.conduct_strategic_analysis(
            business_data, market_data, competitive_data
        )
        
        initiatives = executive_intelligence.generate_strategic_recommendations(
            strategic_analysis, strategic_objectives, {'budget': 200000000}
        )
        
        assert len(initiatives) > 0
        for initiative in initiatives:
            assert hasattr(initiative, 'initiative_id')
            assert hasattr(initiative, 'strategic_priority')
            assert initiative.expected_roi > 0

class TestDataManager:
    """Test data management system"""
    
    @pytest.fixture
    def data_manager(self):
        """Create data manager instance"""
        return DataManager(":memory:")  # Use in-memory database for testing
    
    def test_data_source_registration(self, data_manager):
        """Test data source registration"""
        config = DataIngestionConfig(
            source_id="test_source",
            source_type=DataSource.FILE,
            connection_string="test.csv",
            schedule="0 2 * * *",
            enabled=True
        )
        
        data_manager.register_data_source(config)
        assert config.source_id in data_manager.data_sources
    
    def test_data_ingestion(self, data_manager):
        """Test data ingestion"""
        test_data = pd.DataFrame({
            'id': range(1, 11),
            'value': np.random.uniform(1, 100, 10)
        })
        
        dataset_id = data_manager.ingest_data("test_source", test_data)
        
        assert dataset_id is not None
        assert dataset_id in data_manager.quality_reports
        
        # Retrieve data
        retrieved_data = data_manager.retrieve_data(dataset_id)
        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(test_data)
    
    def test_data_quality_assessment(self, data_manager):
        """Test data quality assessment"""
        test_data = pd.DataFrame({
            'complete_col': range(1, 11),
            'missing_col': [1, 2, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10]
        })
        
        dataset_id = data_manager.ingest_data("test_source", test_data)
        quality_report = data_manager.quality_reports[dataset_id]
        
        assert hasattr(quality_report, 'overall_quality')
        assert hasattr(quality_report, 'completeness_score')
        assert hasattr(quality_report, 'issues')
        assert hasattr(quality_report, 'recommendations')

class TestConfiguration:
    """Test configuration management"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert isinstance(config, AnalyticsConfig)
        assert config.platform_name is not None
        assert config.version is not None
        assert config.environment in ['development', 'production']
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        validation_results = config_manager.validate_config()
        
        assert isinstance(validation_results, dict)
        assert 'ml_confidence_threshold' in validation_results
        assert 'max_concurrent_analyses' in validation_results
        assert all(isinstance(v, bool) for v in validation_results.values())
    
    def test_config_update(self):
        """Test configuration update"""
        config_manager = ConfigManager()
        original_threshold = config_manager.config.ml_confidence_threshold
        
        config_manager.update_config(ml_confidence_threshold=0.8)
        new_threshold = config_manager.config.ml_confidence_threshold
        
        assert new_threshold == 0.8
        assert new_threshold != original_threshold

class TestAnalyticsOrchestrator:
    """Test the analytics orchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return AnalyticsOrchestrator()
    
    @pytest.fixture
    def sample_data_sources(self):
        """Create sample data sources"""
        return {
            "main_data": pd.DataFrame({
                'revenue': np.random.uniform(100000, 1000000, 50),
                'customers': np.random.randint(100, 1000, 50),
                'satisfaction': np.random.uniform(3.0, 5.0, 50)
            }),
            "customer_data": pd.DataFrame({
                'customer_id': range(1, 51),
                'age': np.random.randint(18, 80, 50),
                'gender': np.random.choice(['M', 'F'], 50)
            }),
            "transaction_data": pd.DataFrame({
                'customer_id': np.random.choice(range(1, 51), 200),
                'amount': np.random.uniform(10, 500, 200)
            })
        }
    
    def test_platform_initialization(self, orchestrator):
        """Test platform initialization"""
        assert orchestrator is not None
        assert orchestrator.analytics_engine is not None
        assert orchestrator.predictive_analytics is not None
        assert orchestrator.customer_analytics is not None
        assert len(orchestrator.pipelines) > 0
    
    def test_pipeline_execution(self, orchestrator, sample_data_sources):
        """Test pipeline execution"""
        report = orchestrator.execute_pipeline("customer_intelligence", sample_data_sources)
        
        assert report is not None
        assert hasattr(report, 'report_id')
        assert hasattr(report, 'title')
        assert hasattr(report, 'confidence_score')
        assert 0 <= report.confidence_score <= 1
    
    def test_comprehensive_analysis(self, orchestrator, sample_data_sources):
        """Test comprehensive analysis"""
        reports = orchestrator.execute_comprehensive_analysis(sample_data_sources)
        
        assert isinstance(reports, dict)
        # At least some pipelines should execute successfully
        successful_pipelines = [pid for pid, report in reports.items() if report is not None]
        assert len(successful_pipelines) > 0
    
    def test_platform_status(self, orchestrator):
        """Test platform status retrieval"""
        status = orchestrator.get_platform_status()
        
        assert 'platform_info' in status
        assert 'pipelines' in status
        assert 'performance' in status
        assert status['platform_info']['name'] is not None
        assert len(status['pipelines']) > 0

# Integration test
class TestAnalyticsIntegration:
    """Integration tests for the complete platform"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return AnalyticsOrchestrator()
    
    def test_end_to_end_analysis(self, orchestrator):
        """Test end-to-end analytics analysis"""
        # Create comprehensive test data
        data_sources = {
            "main_data": pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=50, freq='D'),
                'revenue': np.random.uniform(50000, 150000, 50),
                'customers': np.random.randint(100, 500, 50),
                'satisfaction': np.random.uniform(3.5, 5.0, 50)
            }),
            "customer_data": pd.DataFrame({
                'customer_id': range(1, 101),
                'age': np.random.randint(18, 80, 100),
                'gender': np.random.choice(['M', 'F'], 100),
                'income': np.random.randint(30000, 150000, 100)
            }),
            "transaction_data": pd.DataFrame({
                'customer_id': np.random.choice(range(1, 101), 500),
                'date': pd.date_range('2023-01-01', periods=500, freq='H'),
                'amount': np.random.uniform(10, 1000, 500)
            }),
            "operational_data": pd.DataFrame({
                'process_id': range(1, 51),
                'efficiency': np.random.uniform(60, 95, 50),
                'cost': np.random.uniform(1000, 50000, 50)
            }),
            "market_data": pd.DataFrame({
                'market_name': ['Technology Sector'] * 25,
                'revenue': np.random.uniform(10000000, 100000000, 25),
                'growth_rate': np.random.uniform(0.05, 0.25, 25)
            }),
            "competitor_data": pd.DataFrame({
                'id': range(1, 6),
                'name': [f'Competitor {i}' for i in range(1, 6)],
                'revenue': np.random.uniform(1000000, 50000000, 5),
                'market_share': np.random.uniform(0.05, 0.30, 5)
            })
        }
        
        # Run comprehensive analysis
        reports = orchestrator.execute_comprehensive_analysis(data_sources)
        
        # Verify results
        assert len(reports) > 0
        
        for pipeline_id, report in reports.items():
            if report is not None:  # Some pipelines might fail, that's OK
                assert report.report_id is not None
                assert len(report.key_insights) > 0
                assert len(report.recommendations) > 0
                assert 0 <= report.confidence_score <= 1
        
        # Test platform status
        status = orchestrator.get_platform_status()
        assert 'platform_info' in status
        assert 'pipelines' in status
        
        # Test insights summary
        insights_summary = orchestrator.get_insights_summary()
        assert 'total_insights' in insights_summary
        assert insights_summary['total_insights'] > 0

if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])