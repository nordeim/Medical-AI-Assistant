"""
Pipeline Analysis and Management System
Analyzes sales pipeline health, performance, and forecasting
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import logging
from dataclasses import asdict

class PipelineAnalyzer:
    """Sales pipeline analysis and management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pipeline parameters
        self.pipeline_stages = config.get('pipeline_stages', [
            'lead', 'qualified', 'proposal', 'negotiation', 'closed_won', 'closed_lost'
        ])
        self.stage_probabilities = config.get('stage_probabilities', {
            'lead': 0.10,
            'qualified': 0.25,
            'proposal': 0.50,
            'negotiation': 0.75,
            'closed_won': 1.0,
            'closed_lost': 0.0
        })
        self.avg_sales_cycle_days = config.get('avg_sales_cycle_days', 60)
        self.healthy_conversion_rates = config.get('healthy_conversion_rates', {
            'lead_to_qualified': 0.25,
            'qualified_to_proposal': 0.60,
            'proposal_to_negotiation': 0.70,
            'negotiation_to_close': 0.60
        })
    
    def analyze_pipeline(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive pipeline analysis"""
        self.logger.info(f"Analyzing pipeline with {len(pipeline_data)} deals")
        
        if not pipeline_data:
            return {'error': 'No pipeline data provided'}
        
        # Pipeline overview
        pipeline_overview = self._analyze_pipeline_overview(pipeline_data)
        
        # Stage analysis
        stage_analysis = self._analyze_pipeline_stages(pipeline_data)
        
        # Conversion analysis
        conversion_analysis = self._analyze_conversions(pipeline_data)
        
        # Pipeline health
        health_assessment = self._assess_pipeline_health(pipeline_overview, stage_analysis, conversion_analysis)
        
        # Velocity analysis
        velocity_analysis = self._analyze_pipeline_velocity(pipeline_data)
        
        # Risk analysis
        risk_analysis = self._analyze_pipeline_risks(pipeline_data)
        
        # Performance by rep
        rep_performance = self._analyze_rep_performance(pipeline_data)
        
        # Generate insights
        insights = self._generate_pipeline_insights(
            pipeline_overview, stage_analysis, conversion_analysis, health_assessment
        )
        
        return {
            'pipeline_overview': pipeline_overview,
            'stage_analysis': stage_analysis,
            'conversion_analysis': conversion_analysis,
            'health_assessment': health_assessment,
            'velocity_analysis': velocity_analysis,
            'risk_analysis': risk_analysis,
            'rep_performance': rep_performance,
            'insights': insights,
            'generated_at': datetime.now().isoformat()
        }
    
    def forecast_from_pipeline(self, pipeline_data: List[Dict[str, Any]], 
                             forecast_months: int = 6) -> Dict[str, Any]:
        """Generate revenue forecast from pipeline data"""
        self.logger.info(f"Generating {forecast_months}-month pipeline forecast")
        
        # Analyze current pipeline composition
        pipeline_analysis = self.analyze_pipeline(pipeline_data)
        
        # Calculate weighted pipeline value
        weighted_pipeline = pipeline_analysis['pipeline_overview']['total_weighted_value']
        
        # Apply time-based forecasting
        time_forecast = self._forecast_by_time(pipeline_data, forecast_months)
        
        # Apply stage-based forecasting
        stage_forecast = self._forecast_by_stage(pipeline_data, forecast_months)
        
        # Apply rep-based forecasting
        rep_forecast = self._forecast_by_rep(pipeline_data, forecast_months)
        
        # Combine forecasts
        combined_forecast = self._combine_pipeline_forecasts(
            time_forecast, stage_forecast, rep_forecast, weighted_pipeline
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_pipeline_confidence_intervals(
            pipeline_data, combined_forecast
        )
        
        return {
            'forecast_months': forecast_months,
            'monthly_forecasts': combined_forecast,
            'confidence_intervals': confidence_intervals,
            'forecast_methods': {
                'time_based': time_forecast,
                'stage_based': stage_forecast,
                'rep_based': rep_forecast
            },
            'pipeline_coverage': self._calculate_pipeline_coverage(pipeline_data, combined_forecast),
            'forecast_quality': self._assess_forecast_quality(pipeline_data),
            'generated_at': datetime.now().isoformat()
        }
    
    def _analyze_pipeline_overview(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pipeline overview metrics"""
        total_value = sum(Decimal(str(d.get('deal_value', '0'))) for d in pipeline_data)
        
        # Calculate weighted value
        weighted_value = Decimal('0')
        for deal in pipeline_data:
            deal_value = Decimal(str(deal.get('deal_value', '0')))
            stage = deal.get('stage', 'lead')
            probability = self.stage_probabilities.get(stage, 0.1)
            weighted_value += deal_value * Decimal(str(probability))
        
        # Count deals by stage
        deals_by_stage = {}
        for deal in pipeline_data:
            stage = deal.get('stage', 'lead')
            deals_by_stage[stage] = deals_by_stage.get(stage, 0) + 1
        
        # Calculate average deal size
        avg_deal_size = total_value / len(pipeline_data) if pipeline_data else Decimal('0')
        
        # Pipeline velocity (deals per month)
        monthly_deal_velocity = len(pipeline_data) / 12  # Assume 12 months of pipeline
        
        return {
            'total_deals': len(pipeline_data),
            'total_pipeline_value': total_value,
            'total_weighted_value': weighted_value,
            'average_deal_size': avg_deal_size,
            'deals_by_stage': deals_by_stage,
            'pipeline_coverage_ratio': float(weighted_value / self._calculate_monthly_quota()),
            'velocity_deals_per_month': monthly_deal_velocity,
            'pipeline_health_score': self._calculate_pipeline_health_score(pipeline_data)
        }
    
    def _analyze_pipeline_stages(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pipeline by stages"""
        stage_analysis = {}
        
        for stage in self.pipeline_stages:
            stage_deals = [d for d in pipeline_data if d.get('stage') == stage]
            
            if not stage_deals:
                continue
            
            total_value = sum(Decimal(str(d.get('deal_value', '0'))) for d in stage_deals)
            avg_deal_value = total_value / len(stage_deals)
            
            # Calculate average days in stage
            avg_days_in_stage = sum(d.get('days_in_stage', 0) for d in stage_deals) / len(stage_deals)
            
            # Calculate probability and weighted value
            stage_probability = self.stage_probabilities.get(stage, 0)
            weighted_value = total_value * Decimal(str(stage_probability))
            
            # Assess stage health
            stage_health = self._assess_stage_health(stage, stage_deals)
            
            stage_analysis[stage] = {
                'deal_count': len(stage_deals),
                'total_value': total_value,
                'average_deal_value': avg_deal_value,
                'average_days_in_stage': avg_days_in_stage,
                'stage_probability': stage_probability,
                'weighted_value': weighted_value,
                'stage_health': stage_health,
                'conversion_rate_to_next': self._calculate_stage_conversion_rate(stage, stage_deals)
            }
        
        return stage_analysis
    
    def _analyze_conversions(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversion rates between stages"""
        conversion_rates = {}
        
        # Calculate conversion rates for each stage transition
        for i in range(len(self.pipeline_stages) - 1):
            current_stage = self.pipeline_stages[i]
            next_stage = self.pipeline_stages[i + 1]
            
            # Count deals that moved from current to next stage
            current_stage_deals = [d for d in pipeline_data if d.get('stage') == current_stage]
            next_stage_deals = [d for d in pipeline_data if d.get('stage') == next_stage]
            
            # Simplified conversion rate calculation
            if current_stage_deals:
                # In reality, you'd track historical conversions
                conversion_rate = min(len(next_stage_deals) / len(current_stage_deals), 1.0)
            else:
                conversion_rate = self.healthy_conversion_rates.get(
                    f"{current_stage}_to_{next_stage}", 0.5
                )
            
            conversion_rates[f"{current_stage}_to_{next_stage}"] = {
                'rate': conversion_rate,
                'healthy_threshold': self.healthy_conversion_rates.get(
                    f"{current_stage}_to_{next_stage}", 0.5
                ),
                'performance': self._assess_conversion_performance(
                    conversion_rate, 
                    self.healthy_conversion_rates.get(f"{current_stage}_to_{next_stage}", 0.5)
                )
            }
        
        return {
            'conversion_rates': conversion_rates,
            'overall_funnel_health': self._assess_funnel_health(conversion_rates),
            'bottlenecks': self._identify_conversion_bottlenecks(conversion_rates)
        }
    
    def _assess_pipeline_health(self, overview: Dict[str, Any], 
                              stage_analysis: Dict[str, Any], 
                              conversion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall pipeline health"""
        health_score = 0
        health_factors = {}
        
        # Pipeline coverage (30% weight)
        coverage_ratio = overview.get('pipeline_coverage_ratio', 0)
        coverage_score = min(coverage_ratio / 2.0, 1.0) * 30  # 2x coverage is excellent
        health_factors['pipeline_coverage'] = coverage_score
        
        # Stage distribution (25% weight)
        stage_health_scores = []
        for stage, data in stage_analysis.items():
            days_in_stage = data.get('average_days_in_stage', 0)
            
            # Optimal stage durations (in days)
            optimal_durations = {
                'lead': 7, 'qualified': 14, 'proposal': 21, 
                'negotiation': 30, 'closed_won': 7, 'closed_lost': 7
            }
            
            optimal_duration = optimal_durations.get(stage, 14)
            stage_score = max(0, 1 - abs(days_in_stage - optimal_duration) / optimal_duration) * 25
            stage_health_scores.append(stage_score)
        
        avg_stage_score = sum(stage_health_scores) / len(stage_health_scores) if stage_health_scores else 0
        health_factors['stage_distribution'] = avg_stage_score
        
        # Conversion health (25% weight)
        conversion_rates = conversion_analysis.get('conversion_rates', {})
        conversion_scores = []
        for transition, data in conversion_rates.items():
            actual_rate = data.get('rate', 0)
            healthy_rate = data.get('healthy_threshold', 0.5)
            conversion_score = min(actual_rate / healthy_rate, 1.0) * 25
            conversion_scores.append(conversion_score)
        
        avg_conversion_score = sum(conversion_scores) / len(conversion_scores) if conversion_scores else 0
        health_factors['conversion_rates'] = avg_conversion_score
        
        # Pipeline velocity (20% weight)
        velocity_score = min(overview.get('velocity_deals_per_month', 0) / 10, 1.0) * 20  # 10 deals/month is good
        health_factors['pipeline_velocity'] = velocity_score
        
        total_health_score = sum(health_factors.values())
        
        # Overall health rating
        if total_health_score >= 80:
            health_rating = 'Excellent'
        elif total_health_score >= 65:
            health_rating = 'Good'
        elif total_health_score >= 50:
            health_rating = 'Fair'
        elif total_health_score >= 35:
            health_rating = 'Poor'
        else:
            health_rating = 'Critical'
        
        return {
            'overall_health_score': total_health_score,
            'health_rating': health_rating,
            'health_factors': health_factors,
            'health_trend': self._calculate_health_trend(overview, stage_analysis)
        }
    
    def _analyze_pipeline_velocity(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pipeline velocity and timing"""
        velocity_metrics = {}
        
        # Days in pipeline
        days_in_pipeline = [d.get('total_days_in_pipeline', 0) for d in pipeline_data if d.get('total_days_in_pipeline')]
        if days_in_pipeline:
            velocity_metrics['average_days_in_pipeline'] = sum(days_in_pipeline) / len(days_in_pipeline)
            velocity_metrics['median_days_in_pipeline'] = self._calculate_median(days_in_pipeline)
        
        # Days by stage
        stage_velocities = {}
        for stage in self.pipeline_stages:
            stage_deals = [d for d in pipeline_data if d.get('stage') == stage]
            if stage_deals:
                avg_days = sum(d.get('days_in_stage', 0) for d in stage_deals) / len(stage_deals)
                stage_velocities[stage] = avg_days
        
        velocity_metrics['stage_velocities'] = stage_velocities
        
        # Stale deals analysis
        stale_threshold = self.config.get('stale_deal_threshold_days', 30)
        stale_deals = [d for d in pipeline_data if d.get('days_in_stage', 0) > stale_threshold]
        
        velocity_metrics['stale_deals'] = {
            'count': len(stale_deals),
            'percentage': len(stale_deals) / len(pipeline_data) if pipeline_data else 0,
            'total_value': sum(Decimal(str(d.get('deal_value', '0'))) for d in stale_deals)
        }
        
        return velocity_metrics
    
    def _analyze_pipeline_risks(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze pipeline risks and opportunities"""
        risk_analysis = {}
        
        # High-value deals at risk
        high_value_threshold = self.config.get('high_value_threshold', 100000)
        high_value_deals = [
            d for d in pipeline_data 
            if Decimal(str(d.get('deal_value', '0'))) >= Decimal(str(high_value_threshold))
        ]
        
        # Stale deals (high risk)
        stale_deals = [
            d for d in pipeline_data 
            if d.get('days_in_stage', 0) > self.config.get('stale_deal_threshold_days', 30)
        ]
        
        # Low probability deals
        low_probability_deals = [
            d for d in pipeline_data 
            if self.stage_probabilities.get(d.get('stage', ''), 0) < 0.3
        ]
        
        # Deals with no recent activity
        inactive_threshold = self.config.get('inactive_deal_threshold_days', 14)
        inactive_deals = [
            d for d in pipeline_data 
            if d.get('days_since_last_activity', 0) > inactive_threshold
        ]
        
        risk_analysis = {
            'high_value_risk': {
                'count': len(high_value_deals),
                'total_value': sum(Decimal(str(d.get('deal_value', '0'))) for d in high_value_deals),
                'percentage_of_pipeline': len(high_value_deals) / len(pipeline_data) if pipeline_data else 0
            },
            'stale_deals_risk': {
                'count': len(stale_deals),
                'total_value': sum(Decimal(str(d.get('deal_value', '0'))) for d in stale_deals),
                'risk_level': 'High' if len(stale_deals) / len(pipeline_data) > 0.2 else 'Medium'
            },
            'low_probability_risk': {
                'count': len(low_probability_deals),
                'total_value': sum(Decimal(str(d.get('deal_value', '0'))) for d in low_probability_deals),
                'risk_level': 'Medium'
            },
            'inactive_deals_risk': {
                'count': len(inactive_deals),
                'total_value': sum(Decimal(str(d.get('deal_value', '0'))) for d in inactive_deals),
                'risk_level': 'Medium'
            }
        }
        
        # Overall risk assessment
        total_risk_deals = len(high_value_deals) + len(stale_deals) + len(low_probability_deals) + len(inactive_deals)
        risk_percentage = total_risk_deals / len(pipeline_data) if pipeline_data else 0
        
        if risk_percentage > 0.4:
            overall_risk = 'High'
        elif risk_percentage > 0.25:
            overall_risk = 'Medium'
        else:
            overall_risk = 'Low'
        
        risk_analysis['overall_risk_assessment'] = {
            'risk_level': overall_risk,
            'deals_at_risk': total_risk_deals,
            'percentage_at_risk': risk_percentage
        }
        
        return risk_analysis
    
    def _analyze_rep_performance(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sales representative performance"""
        rep_performance = {}
        
        # Group deals by sales rep
        deals_by_rep = {}
        for deal in pipeline_data:
            rep = deal.get('sales_rep', 'Unknown')
            if rep not in deals_by_rep:
                deals_by_rep[rep] = []
            deals_by_rep[rep].append(deal)
        
        # Calculate performance metrics for each rep
        for rep, deals in deals_by_rep.items():
            total_value = sum(Decimal(str(d.get('deal_value', '0'))) for d in deals)
            weighted_value = sum(
                Decimal(str(d.get('deal_value', '0'))) * 
                Decimal(str(self.stage_probabilities.get(d.get('stage', ''), 0.1)))
                for d in deals
            )
            
            # Average deal size
            avg_deal_size = total_value / len(deals) if deals else Decimal('0')
            
            # Pipeline distribution
            stage_distribution = {}
            for deal in deals:
                stage = deal.get('stage', 'lead')
                stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            
            # Deal velocity (deals per month)
            velocity = len(deals) / 12  # Assume 12 months of activity
            
            # Risk assessment
            stale_deals = len([d for d in deals if d.get('days_in_stage', 0) > 30])
            
            rep_performance[rep] = {
                'deal_count': len(deals),
                'total_pipeline_value': total_value,
                'weighted_pipeline_value': weighted_value,
                'average_deal_size': avg_deal_size,
                'stage_distribution': stage_distribution,
                'velocity_deals_per_month': velocity,
                'stale_deals_count': stale_deals,
                'performance_score': self._calculate_rep_performance_score(deals),
                'risk_score': min(stale_deals / len(deals), 1.0) if deals else 0
            }
        
        return rep_performance
    
    def _generate_pipeline_insights(self, overview: Dict[str, Any], 
                                  stage_analysis: Dict[str, Any], 
                                  conversion_analysis: Dict[str, Any],
                                  health_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable pipeline insights"""
        insights = []
        
        # Pipeline health insights
        health_rating = health_assessment.get('health_rating', 'Unknown')
        if health_rating in ['Poor', 'Critical']:
            insights.append({
                'type': 'warning',
                'category': 'Pipeline Health',
                'insight': f'Pipeline health is {health_rating} (Score: {health_assessment.get("overall_health_score", 0):.1f})',
                'recommendation': 'Focus on pipeline quality and conversion optimization',
                'priority': 'High'
            })
        
        # Coverage insights
        coverage_ratio = overview.get('pipeline_coverage_ratio', 0)
        if coverage_ratio < 1.5:
            insights.append({
                'type': 'warning',
                'category': 'Pipeline Coverage',
                'insight': f'Pipeline coverage is low at {coverage_ratio:.1f}x quota',
                'recommendation': 'Increase prospecting efforts to build pipeline',
                'priority': 'High'
            })
        elif coverage_ratio > 4:
            insights.append({
                'type': 'info',
                'category': 'Pipeline Coverage',
                'insight': f'Excellent pipeline coverage at {coverage_ratio:.1f}x quota',
                'recommendation': 'Focus on converting existing pipeline',
                'priority': 'Medium'
            })
        
        # Conversion insights
        bottlenecks = conversion_analysis.get('bottlenecks', [])
        if bottlenecks:
            insights.append({
                'type': 'warning',
                'category': 'Conversion',
                'insight': f'Conversion bottlenecks detected: {", ".join(bottlenecks)}',
                'recommendation': 'Focus improvement efforts on identified bottlenecks',
                'priority': 'High'
            })
        
        # Velocity insights
        stale_deals_pct = overview.get('velocity_deals_per_month', 0)  # Simplified
        if stale_deals_pct > 0.2:  # More than 20% stale deals
            insights.append({
                'type': 'warning',
                'category': 'Pipeline Velocity',
                'insight': 'High percentage of stale deals in pipeline',
                'recommendation': 'Review and re-engage stale deals or remove from pipeline',
                'priority': 'Medium'
            })
        
        return insights
    
    # Helper methods
    
    def _calculate_monthly_quota(self) -> Decimal:
        """Calculate monthly revenue quota"""
        # This would typically come from sales targets
        # For demo, using a reasonable default
        return Decimal('1000000')
    
    def _calculate_pipeline_health_score(self, pipeline_data: List[Dict[str, Any]]) -> float:
        """Calculate pipeline health score"""
        if not pipeline_data:
            return 0
        
        # Simplified health calculation
        active_deals = len([d for d in pipeline_data if d.get('stage') not in ['closed_won', 'closed_lost']])
        total_deals = len(pipeline_data)
        
        activity_score = active_deals / total_deals if total_deals > 0 else 0
        
        # Additional factors would be included in full implementation
        return activity_score * 100
    
    def _assess_stage_health(self, stage: str, stage_deals: List[Dict[str, Any]]) -> str:
        """Assess health of a specific stage"""
        if not stage_deals:
            return 'Unknown'
        
        avg_days = sum(d.get('days_in_stage', 0) for d in stage_deals) / len(stage_deals)
        
        # Optimal stage durations
        optimal_durations = {
            'lead': 7, 'qualified': 14, 'proposal': 21, 
            'negotiation': 30, 'closed_won': 7, 'closed_lost': 7
        }
        
        optimal_duration = optimal_durations.get(stage, 14)
        variance = abs(avg_days - optimal_duration) / optimal_duration
        
        if variance <= 0.2:
            return 'Healthy'
        elif variance <= 0.5:
            return 'Moderate'
        else:
            return 'Poor'
    
    def _calculate_stage_conversion_rate(self, stage: str, stage_deals: List[Dict[str, Any]]) -> float:
        """Calculate conversion rate from current stage"""
        # This would require historical data in a real implementation
        # For demo, return expected conversion rate
        conversion_rates = {
            'lead': 0.25, 'qualified': 0.60, 'proposal': 0.70, 
            'negotiation': 0.60, 'closed_won': 1.0, 'closed_lost': 0.0
        }
        
        return conversion_rates.get(stage, 0.5)
    
    def _assess_conversion_performance(self, actual_rate: float, healthy_rate: float) -> str:
        """Assess conversion rate performance"""
        if actual_rate >= healthy_rate * 1.1:
            return 'Excellent'
        elif actual_rate >= healthy_rate * 0.9:
            return 'Good'
        elif actual_rate >= healthy_rate * 0.7:
            return 'Fair'
        else:
            return 'Poor'
    
    def _assess_funnel_health(self, conversion_rates: Dict[str, Any]) -> str:
        """Assess overall funnel health"""
        if not conversion_rates:
            return 'Unknown'
        
        good_conversions = 0
        total_conversions = len(conversion_rates)
        
        for conversion_data in conversion_rates.values():
            if conversion_data.get('performance') in ['Excellent', 'Good']:
                good_conversions += 1
        
        ratio = good_conversions / total_conversions if total_conversions > 0 else 0
        
        if ratio >= 0.8:
            return 'Excellent'
        elif ratio >= 0.6:
            return 'Good'
        elif ratio >= 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    def _identify_conversion_bottlenecks(self, conversion_rates: Dict[str, Any]) -> List[str]:
        """Identify conversion bottlenecks"""
        bottlenecks = []
        
        for transition, data in conversion_rates.items():
            if data.get('performance') in ['Poor']:
                bottlenecks.append(transition.replace('_to_', ' → '))
        
        return bottlenecks
    
    def _calculate_health_trend(self, overview: Dict[str, Any], stage_analysis: Dict[str, Any]) -> str:
        """Calculate pipeline health trend"""
        # Simplified trend calculation
        # In reality, this would compare current health to historical data
        return 'stable'  # Placeholder
    
    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median value"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def _forecast_by_time(self, pipeline_data: List[Dict[str, Any]], months: int) -> List[Decimal]:
        """Forecast revenue based on time factors"""
        # Time-based forecasting considers expected close dates
        monthly_forecasts = [Decimal('0')] * months
        today = date.today()
        
        for deal in pipeline_data:
            expected_close = deal.get('expected_close_date')
            if isinstance(expected_close, str):
                expected_close = datetime.strptime(expected_close, '%Y-%m-%d').date()
            
            if expected_close and expected_close > today:
                deal_value = Decimal(str(deal.get('deal_value', '0')))
                stage = deal.get('stage', 'lead')
                probability = self.stage_probabilities.get(stage, 0.1)
                
                # Calculate months until close
                months_until_close = max(0, (expected_close.year - today.year) * 12 + 
                                       (expected_close.month - today.month))
                
                if months_until_close < months:
                    # Distribute weighted value across the forecast period
                    monthly_value = deal_value * Decimal(str(probability)) / months_until_close if months_until_close > 0 else Decimal('0')
                    
                    for month_idx in range(months_until_close, months):
                        monthly_forecasts[month_idx] += monthly_value
        
        return monthly_forecasts
    
    def _forecast_by_stage(self, pipeline_data: List[Dict[str, Any]], months: int) -> List[Decimal]:
        """Forecast revenue based on stage progression"""
        # Stage-based forecasting assumes deals progress through stages
        monthly_forecasts = [Decimal('0')] * months
        
        # Simplified stage progression model
        stage_progression = {
            'lead': 7, 'qualified': 14, 'proposal': 21, 'negotiation': 30
        }
        
        for deal in pipeline_data:
            if deal.get('stage') in ['closed_won', 'closed_lost']:
                continue
            
            deal_value = Decimal(str(deal.get('deal_value', '0')))
            current_stage = deal.get('stage', 'lead')
            days_in_stage = deal.get('days_in_stage', 0)
            
            # Calculate expected progression time
            progression_days = stage_progression.get(current_stage, 21)
            remaining_days = max(0, progression_days - days_in_stage)
            
            # Distribute forecast over remaining months
            if remaining_days > 0:
                stage_probability = self.stage_probabilities.get(current_stage, 0.1)
                monthly_value = deal_value * Decimal(str(stage_probability)) / (remaining_days / 30)
                
                month_idx = min(months - 1, int(remaining_days / 30))
                monthly_forecasts[month_idx] += monthly_value
        
        return monthly_forecasts
    
    def _forecast_by_rep(self, pipeline_data: List[Dict[str, Any]], months: int) -> List[Decimal]:
        """Forecast revenue based on rep performance"""
        # Rep-based forecasting considers historical performance
        rep_performance = self._analyze_rep_performance(pipeline_data)
        monthly_forecasts = [Decimal('0')] * months
        
        # Calculate average monthly closure rate per rep
        total_weighted_pipeline = sum(
            rep_data.get('weighted_pipeline_value', Decimal('0')) 
            for rep_data in rep_performance.values()
        )
        
        for rep, performance_data in rep_performance.items():
            rep_weighted_value = performance_data.get('weighted_pipeline_value', Decimal('0'))
            rep_monthly_closure = rep_weighted_value / months  # Simplified
            
            for month_idx in range(months):
                monthly_forecasts[month_idx] += rep_monthly_closure
        
        return monthly_forecasts
    
    def _combine_pipeline_forecasts(self, time_forecast: List[Decimal], 
                                  stage_forecast: List[Decimal], 
                                  rep_forecast: List[Decimal],
                                  total_weighted_value: Decimal) -> List[Decimal]:
        """Combine different forecasting methods"""
        # Weight the forecasts (simplified combination)
        time_weight = 0.4
        stage_weight = 0.3
        rep_weight = 0.3
        
        combined_forecast = []
        
        for i in range(len(time_forecast)):
            combined_value = (
                time_forecast[i] * time_weight +
                stage_forecast[i] * stage_weight +
                rep_forecast[i] * rep_weight
            )
            combined_forecast.append(combined_value)
        
        # Ensure total forecast doesn't exceed weighted pipeline value
        total_forecast = sum(combined_forecast)
        if total_forecast > total_weighted_value:
            scaling_factor = float(total_weighted_value / total_forecast)
            combined_forecast = [value * Decimal(str(scaling_factor)) for value in combined_forecast]
        
        return combined_forecast
    
    def _calculate_pipeline_confidence_intervals(self, pipeline_data: List[Dict[str, Any]], 
                                               forecast: List[Decimal]) -> Dict[str, List[Decimal]]:
        """Calculate confidence intervals for pipeline forecast"""
        # Simplified confidence intervals based on pipeline quality
        base_confidence = 0.75  # Base confidence level
        
        confidence_multipliers = {
            'p10': 1.645,  # 90% confidence
            'p25': 1.281,  # 80% confidence
            'p75': 1.281,  # 20% confidence (lower bound)
            'p90': 1.645   # 10% confidence (lower bound)
        }
        
        intervals = {}
        
        for interval_name, multiplier in confidence_multipliers.items():
            interval_values = []
            for value in forecast:
                error_margin = value * (1 - base_confidence) * multiplier
                if interval_name in ['p10', 'p25']:  # Upper bounds
                    interval_values.append(value + error_margin)
                else:  # Lower bounds
                    interval_values.append(max(value - error_margin, Decimal('0')))
            intervals[interval_name] = interval_values
        
        return intervals
    
    def _calculate_pipeline_coverage(self, pipeline_data: List[Dict[str, Any]], forecast: List[Decimal]) -> Dict[str, float]:
        """Calculate pipeline coverage metrics"""
        total_weighted_pipeline = sum(
            Decimal(str(d.get('deal_value', '0'))) * 
            Decimal(str(self.stage_probabilities.get(d.get('stage', ''), 0.1)))
            for d in pipeline_data
        )
        
        monthly_quota = self._calculate_monthly_quota()
        
        coverage_ratios = []
        for month_forecast in forecast:
            coverage_ratio = float(total_weighted_pipeline / (monthly_quota * month_forecast)) if month_forecast > 0 else 0
            coverage_ratios.append(coverage_ratio)
        
        return {
            'coverage_ratios': coverage_ratios,
            'average_coverage': sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0,
            'target_coverage': 3.0  # Target 3x coverage
        }
    
    def _assess_forecast_quality(self, pipeline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess quality of pipeline forecast"""
        # Calculate forecast quality based on pipeline characteristics
        quality_score = 0
        
        # Pipeline size (30% weight)
        if len(pipeline_data) > 20:
            quality_score += 30
        elif len(pipeline_data) > 10:
            quality_score += 20
        else:
            quality_score += 10
        
        # Pipeline distribution (40% weight)
        stage_distribution = {}
        for deal in pipeline_data:
            stage = deal.get('stage', 'lead')
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
        
        # Healthy distribution has deals in multiple stages
        if len(stage_distribution) >= 4:
            quality_score += 40
        elif len(stage_distribution) >= 3:
            quality_score += 30
        else:
            quality_score += 20
        
        # Deal freshness (30% weight)
        fresh_deals = len([d for d in pipeline_data if d.get('days_in_stage', 0) < 14])
        freshness_ratio = fresh_deals / len(pipeline_data) if pipeline_data else 0
        
        quality_score += freshness_ratio * 30
        
        # Quality rating
        if quality_score >= 90:
            quality = 'Excellent'
        elif quality_score >= 75:
            quality = 'Good'
        elif quality_score >= 60:
            quality = 'Fair'
        else:
            quality = 'Poor'
        
        return {
            'quality_score': quality_score,
            'quality_rating': quality,
            'confidence_level': quality_score / 100
        }
    
    def _calculate_rep_performance_score(self, deals: List[Dict[str, Any]]) -> float:
        """Calculate performance score for sales representative"""
        if not deals:
            return 0
        
        score = 0
        
        # Deal count (25% weight)
        deal_count_score = min(len(deals) / 20, 1) * 25  # 20 deals is good
        score += deal_count_score
        
        # Weighted pipeline value (35% weight)
        total_weighted_value = sum(
            Decimal(str(d.get('deal_value', '0'))) * 
            Decimal(str(self.stage_probabilities.get(d.get('stage', ''), 0.1)))
            for d in deals
        )
        pipeline_score = min(float(total_weighted_value / 1000000), 1) * 35  # $1M is good
        score += pipeline_score
        
        # Deal quality (40% weight)
        # Higher stage probability = better quality
        avg_stage_probability = sum(
            self.stage_probabilities.get(d.get('stage', ''), 0.1) for d in deals
        ) / len(deals)
        
        quality_score = avg_stage_probability * 40
        score += quality_score
        
        return min(score, 100)