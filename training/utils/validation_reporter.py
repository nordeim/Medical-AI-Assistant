"""Validation reporting utilities for generating comprehensive data quality reports."""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from .data_validator import ValidationResult, ValidationConfig


class ValidationReporter:
    """Generate comprehensive validation reports."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize the validation reporter."""
        self.config = config or ValidationConfig()
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self._ensure_template_dir()
        
        # Set up matplotlib for headless operation
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
    
    def _ensure_template_dir(self):
        """Ensure template directory exists."""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default HTML templates."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Validation Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007acc;
        }
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            margin: 20px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        .score-excellent { background-color: #28a745; }
        .score-good { background-color: #17a2b8; }
        .score-fair { background-color: #ffc107; }
        .score-poor { background-color: #dc3545; }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007acc;
        }
        .section h2 {
            margin-top: 0;
            color: #007acc;
        }
        .error { border-left-color: #dc3545; background-color: #f8f9fa; }
        .warning { border-left-color: #ffc107; background-color: #fffbf0; }
        .success { border-left-color: #28a745; background-color: #f0f8f0; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }
        .metric-card h3 {
            margin-top: 0;
            color: #495057;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .recommendations {
            background-color: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 6px;
            padding: 20px;
        }
        .recommendations h3 {
            color: #004085;
            margin-top: 0;
        }
        .recommendations ul {
            margin: 0;
            padding-left: 20px;
        }
        .recommendations li {
            margin-bottom: 8px;
        }
        .recommendations.urgent {
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .recommendations.urgent h3 {
            color: #721c24;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Data Validation Report</h1>
            <h2>{{ dataset_name }}</h2>
            <div class="score-circle {{ score_class }}">
                {{ "%.1f"|format(validation_score) }}%
            </div>
            <p><strong>Validation Status:</strong> 
                {% if validation_result.is_valid %}
                    <span style="color: #28a745;">✓ PASSED</span>
                {% else %}
                    <span style="color: #dc3545;">✗ FAILED</span>
                {% endif %}
            </p>
        </div>

        {% if validation_result.errors %}
        <div class="section error">
            <h2>🚨 Critical Errors ({{ validation_result.errors|length }})</h2>
            <ul>
                {% for error in validation_result.errors %}
                <li>{{ error }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if validation_result.warnings %}
        <div class="section warning">
            <h2>⚠️ Warnings ({{ validation_result.warnings|length }})</h2>
            <ul>
                {% for warning in validation_result.warnings %}
                <li>{{ warning }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if validation_result.metrics %}
        <div class="section success">
            <h2>📊 Quality Metrics</h2>
            
            {% if distribution_charts %}
            <div class="chart-container">
                <h3>Data Distributions</h3>
                {% for chart in distribution_charts %}
                <img src="data:image/png;base64,{{ chart }}" alt="Distribution Chart">
                {% endfor %}
            </div>
            {% endif %}

            <div class="metrics-grid">
                {% for category, metrics in validation_result.metrics.items() %}
                {% if category not in ['distribution'] %}
                <div class="metric-card">
                    <h3>{{ category.replace('_', ' ').title() }}</h3>
                    {% for metric, value in metrics.items() %}
                    <p><strong>{{ metric.replace('_', ' ').title() }}:</strong> 
                        {% if value is number %}
                            {{ "%.2f"|format(value) }}
                        {% else %}
                            {{ value }}
                        {% endif %}
                    </p>
                    {% endfor %}
                </div>
                {% endif %}
                {% endfor %}
            </div>
        </div>
        {% endif %}

        {% if recommendations %}
        <div class="recommendations {{ 'urgent' if has_critical_recommendations else '' }}">
            <h3>📋 Actionable Recommendations</h3>
            <ul>
                {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}

        <div class="timestamp">
            <p>Report generated on {{ timestamp }}</p>
            <p>Validation ID: {{ validation_id }}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(os.path.join(self.template_dir, 'report.html'), 'w') as f:
            f.write(html_template)
    
    def generate_html_report(self, validation_result: ValidationResult, 
                           output_path: str, dataset_name: str = "Training Dataset",
                           data_summary: Dict[str, Any] = None) -> str:
        """Generate comprehensive HTML report."""
        # Load template
        template_path = os.path.join(self.template_dir, 'report.html')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                template = Template(f.read())
        else:
            # Fallback to embedded template
            template = Template(self._get_default_template())
        
        # Prepare chart data
        distribution_charts = []
        if 'distribution' in validation_result.metrics:
            distribution_charts = self._create_distribution_charts(validation_result.metrics['distribution'])
        
        # Generate recommendations
        recommendations, has_critical = self._generate_recommendations(validation_result)
        
        # Calculate score class
        score = validation_result.score * 100
        if score >= 90:
            score_class = "score-excellent"
        elif score >= 80:
            score_class = "score-good"
        elif score >= 70:
            score_class = "score-fair"
        else:
            score_class = "score-poor"
        
        # Generate validation ID
        validation_id = hashlib.md5(f"{validation_result.timestamp.isoformat()}{dataset_name}".encode()).hexdigest()[:8]
        
        # Render template
        html_content = template.render(
            validation_result=validation_result,
            validation_score=score,
            score_class=score_class,
            dataset_name=dataset_name,
            distribution_charts=distribution_charts,
            recommendations=recommendations,
            has_critical_recommendations=has_critical,
            timestamp=validation_result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            validation_id=validation_id
        )
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_json_report(self, validation_result: ValidationResult, 
                           output_path: str, data_summary: Dict[str, Any] = None) -> str:
        """Generate JSON summary report for automated processing."""
        report_data = {
            'metadata': {
                'timestamp': validation_result.timestamp.isoformat(),
                'validation_id': hashlib.md5(f"{validation_result.timestamp.isoformat()}json".encode()).hexdigest()[:8],
                'generator': 'DataValidator v1.0',
                'data_summary': data_summary or {}
            },
            'summary': {
                'is_valid': validation_result.is_valid,
                'score': validation_result.score,
                'errors_count': len(validation_result.errors),
                'warnings_count': len(validation_result.warnings),
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            },
            'detailed_metrics': validation_result.metrics,
            'recommendations': {
                'urgent': [r for r in self._generate_recommendations(validation_result)[0] if 'critical' in r.lower() or 'error' in r.lower()],
                'standard': [r for r in self._generate_recommendations(validation_result)[0] if not ('critical' in r.lower() or 'error' in r.lower())]
            }
        }
        
        # Write JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_path
    
    def generate_csv_summary(self, validation_result: ValidationResult, 
                           output_path: str, record_count: int) -> str:
        """Generate CSV summary for spreadsheet analysis."""
        summary_data = {
            'Metric': [
                'Overall Score',
                'Validation Status',
                'Total Errors',
                'Total Warnings',
                'Record Count',
                'Timestamp',
                'Data Quality Grade'
            ],
            'Value': [
                f"{validation_result.score:.3f}",
                'PASS' if validation_result.is_valid else 'FAIL',
                len(validation_result.errors),
                len(validation_result.warnings),
                record_count,
                validation_result.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                self._get_quality_grade(validation_result.score)
            ]
        }
        
        # Add key metrics if available
        if 'quality' in validation_result.metrics:
            quality = validation_result.metrics['quality']
            if 'avg_text_quality' in quality:
                summary_data['Metric'].append('Average Text Quality')
                summary_data['Value'].append(f"{quality['avg_text_quality']:.3f}")
            
            if 'coherence' in quality and 'avg_coherence' in quality['coherence']:
                summary_data['Metric'].append('Average Coherence')
                summary_data['Value'].append(f"{quality['coherence']['avg_coherence']:.3f}")
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _create_distribution_charts(self, distributions: Dict[str, Any]) -> List[str]:
        """Create distribution charts and return as base64 encoded images."""
        charts = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for column, stats in distributions.items():
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create a simple visualization of the statistics
                metrics = ['mean', 'median', 'std']
                values = [stats.get(metric, 0) for metric in metrics]
                
                # Create bar chart
                bars = ax.bar(metrics, values, alpha=0.7)
                
                # Color bars based on values
                colors = ['#28a745', '#17a2b8', '#ffc107']
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                
                ax.set_title(f'Distribution Statistics: {column}')
                ax.set_ylabel('Value')
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    ax.text(i, v + max(values) * 0.01, f'{v:.2f}', 
                           ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                charts.append(image_base64)
                
                plt.close()
                
            except Exception as e:
                print(f"Error creating chart for {column}: {e}")
                continue
        
        return charts
    
    def _generate_recommendations(self, validation_result: ValidationResult) -> tuple[List[str], bool]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        has_critical = False
        
        # Error-based recommendations
        if validation_result.errors:
            has_critical = True
            recommendations.append("CRITICAL: Address all validation errors before proceeding with training")
            
            for error in validation_result.errors:
                if "missing required fields" in error.lower():
                    recommendations.append("Ensure all required fields are present in the dataset")
                elif "invalid triage" in error.lower():
                    recommendations.append("Review and standardize triage level classifications")
                elif "invalid age" in error.lower():
                    recommendations.append("Validate age ranges and remove invalid entries")
        
        # Warning-based recommendations
        if validation_result.warnings:
            for warning in validation_result.warnings:
                if "duplicate" in warning.lower():
                    recommendations.append("Remove or merge duplicate records to improve data quality")
                elif "class imbalance" in warning.lower():
                    recommendations.append("Consider rebalancing classes or applying sampling techniques")
                elif "missing data" in warning.lower():
                    recommendations.append("Implement data imputation or collect missing information")
                elif "text quality" in warning.lower():
                    recommendations.append("Review and improve text quality through preprocessing")
                elif "encoding" in warning.lower():
                    recommendations.append("Fix encoding issues by standardizing text format")
                elif "phi" in warning.lower():
                    recommendations.append("Review and remove or anonymize Protected Health Information")
        
        # Quality-based recommendations
        if 'quality' in validation_result.metrics:
            quality = validation_result.metrics['quality']
            
            if 'avg_text_quality' in quality and quality['avg_text_quality'] < 0.7:
                recommendations.append("Improve text quality by enhancing descriptions and structure")
            
            if 'coherence' in quality and 'avg_coherence' in quality['coherence']:
                if quality['coherence']['avg_coherence'] < 0.6:
                    recommendations.append("Enhance conversation coherence between user inputs and responses")
            
            if 'medical_accuracy' in quality:
                med_accuracy = quality['medical_accuracy']
                if 'avg_medical_term_usage' in med_accuracy and med_accuracy['avg_medical_term_usage'] < 0.3:
                    recommendations.append("Increase appropriate medical terminology usage for better context")
        
        # Statistical recommendations
        if 'missing_patterns' in validation_result.metrics:
            missing_pct = validation_result.metrics['missing_patterns'].get('total_missing_percentage', 0)
            if missing_pct > 15:
                recommendations.append("High missing data percentage - consider data collection improvements")
        
        if 'outliers' in validation_result.metrics and validation_result.metrics['outliers']:
            recommendations.append("Review and handle outliers appropriately for your use case")
        
        if 'class_balance' in validation_result.metrics:
            imbalance = validation_result.metrics['class_balance'].get('imbalance_ratio', 1)
            if imbalance > 3.0:
                recommendations.append("Address significant class imbalance through resampling or weighting")
        
        # General recommendations if score is low
        if validation_result.score < 0.8:
            recommendations.append("Overall data quality needs improvement before training")
            has_critical = True
        elif validation_result.score < 0.9:
            recommendations.append("Consider minor data quality improvements for optimal training results")
        
        # Positive recommendations if score is high
        if validation_result.score >= 0.95 and not validation_result.errors:
            recommendations.append("Excellent data quality! Consider this dataset for production training")
        
        return recommendations, has_critical
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.75:
            return "C+"
        elif score >= 0.70:
            return "C"
        elif score >= 0.65:
            return "D+"
        elif score >= 0.60:
            return "D"
        else:
            return "F"
    
    def _get_default_template(self) -> str:
        """Fallback HTML template if file template is not available."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Data Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; }
        .success { background-color: #d1ecf1; border-color: #bee5eb; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Validation Report</h1>
        <h2>{{ dataset_name }}</h2>
        <p>Overall Score: {{ "%.1f"|format(validation_score * 100) }}%</p>
    </div>

    {% if validation_result.errors %}
    <div class="section error">
        <h3>Errors ({{ validation_result.errors|length }})</h3>
        <ul>{% for error in validation_result.errors %}<li>{{ error }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    {% if validation_result.warnings %}
    <div class="section warning">
        <h3>Warnings ({{ validation_result.warnings|length }})</h3>
        <ul>{% for warning in validation_result.warnings %}<li>{{ warning }}</li>{% endfor %}</ul>
    </div>
    {% endif %}

    {% if recommendations %}
    <div class="section success">
        <h3>Recommendations</h3>
        <ul>{% for rec in recommendations %}<li>{{ rec }}</li>{% endfor %}</ul>
    </div>
    {% endif %}
</body>
</html>
"""


class BatchValidationReporter(ValidationReporter):
    """Generate batch validation reports."""
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
    
    def generate_batch_summary_report(self, validation_results: List[ValidationResult],
                                    output_path: str, dataset_names: List[str] = None) -> str:
        """Generate a summary report for multiple validation results."""
        # Calculate aggregate statistics
        total_datasets = len(validation_results)
        passed_datasets = sum(1 for r in validation_results if r.is_valid)
        failed_datasets = total_datasets - passed_datasets
        
        avg_score = np.mean([r.score for r in validation_results])
        total_errors = sum(len(r.errors) for r in validation_results)
        total_warnings = sum(len(r.warnings) for r in validation_results)
        
        # Generate batch report HTML
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Validation Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007acc; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; border: 1px solid #dee2e6; }
        .stat-number { font-size: 24px; font-weight: bold; color: #007acc; }
        .dataset-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .dataset-table th, .dataset-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .dataset-table th { background-color: #f8f9fa; font-weight: 600; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .chart-container { margin: 30px 0; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Batch Validation Summary</h1>
            <p>Generated on {{ timestamp }}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ total_datasets }}</div>
                <div>Total Datasets</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ passed_datasets }}</div>
                <div>Passed Validation</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ failed_datasets }}</div>
                <div>Failed Validation</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ "%.1f"|format(avg_score * 100) }}%</div>
                <div>Average Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_errors }}</div>
                <div>Total Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ total_warnings }}</div>
                <div>Total Warnings</div>
            </div>
        </div>

        {% if distribution_chart %}
        <div class="chart-container">
            <h3>Score Distribution</h3>
            <img src="data:image/png;base64,{{ distribution_chart }}" alt="Score Distribution">
        </div>
        {% endif %}

        <h3>Dataset Details</h3>
        <table class="dataset-table">
            <thead>
                <tr>
                    <th>Dataset Name</th>
                    <th>Validation Status</th>
                    <th>Score</th>
                    <th>Errors</th>
                    <th>Warnings</th>
                </tr>
            </thead>
            <tbody>
                {% for i, result in enumerate(validation_results) %}
                <tr>
                    <td>{{ dataset_names[i] if dataset_names else "Dataset " + (i+1)|string }}</td>
                    <td class="{{ 'pass' if result.is_valid else 'fail' }}">
                        {{ 'PASS' if result.is_valid else 'FAIL' }}
                    </td>
                    <td>{{ "%.1f"|format(result.score * 100) }}%</td>
                    <td>{{ len(result.errors) }}</td>
                    <td>{{ len(result.warnings) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """)
        
        # Create distribution chart for batch
        distribution_chart = self._create_batch_score_chart(validation_results)
        
        # Render template
        html_content = template.render(
            total_datasets=total_datasets,
            passed_datasets=passed_datasets,
            failed_datasets=failed_datasets,
            avg_score=avg_score,
            total_errors=total_errors,
            total_warnings=total_warnings,
            distribution_chart=distribution_chart,
            validation_results=validation_results,
            dataset_names=dataset_names,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _create_batch_score_chart(self, validation_results: List[ValidationResult]) -> Optional[str]:
        """Create a chart showing the distribution of scores across datasets."""
        try:
            scores = [r.score * 100 for r in validation_results]  # Convert to percentage
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create histogram
            ax.hist(scores, bins=10, alpha=0.7, color='#007acc', edgecolor='black')
            ax.set_xlabel('Validation Score (%)')
            ax.set_ylabel('Number of Datasets')
            ax.set_title('Validation Score Distribution Across Datasets')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_score = np.mean(scores)
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}%')
            ax.legend()
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating batch score chart: {e}")
            return None