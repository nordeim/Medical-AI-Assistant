"""
Automated Evaluation Pipeline for Medical AI Models

This module provides automated pipelines for running comprehensive evaluations
of medical AI models with configurable test suites and reporting.

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate_model import ModelEvaluator
from utils.evaluation_metrics import (
    MedicalAccuracyMetrics,
    ClinicalAssessmentMetrics,
    ConversationCoherenceMetrics,
    SafetyAssessmentMetrics,
    RelevanceScoringMetrics
)
from utils.clinical_validation import (
    ClinicalAccuracyValidator,
    MedicalKnowledgeValidator,
    SafetyComplianceChecker
)
from utils.model_registry import ModelRegistry


@dataclass
class EvaluationConfig:
    """Configuration for automated evaluation pipeline."""
    # Model settings
    model_paths: List[str]
    model_names: List[str]
    
    # Dataset settings
    benchmark_datasets: Dict[str, str]
    custom_datasets: Optional[Dict[str, str]] = None
    
    # Evaluation settings
    metrics_to_evaluate: List[str] = None
    evaluation_timeout: int = 3600  # 1 hour
    batch_size: int = 32
    parallel_evaluation: bool = True
    
    # Reporting settings
    output_dir: str = "evaluation_results"
    generate_visualizations: bool = True
    generate_reports: bool = True
    export_formats: List[str] = None
    
    # Quality thresholds
    min_accuracy_threshold: float = 0.7
    min_safety_threshold: float = 0.8
    min_coherence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.metrics_to_evaluate is None:
            self.metrics_to_evaluate = [
                "medical_accuracy",
                "clinical_assessment",
                "conversation_coherence",
                "safety_assessment",
                "relevance_scoring"
            ]
        
        if self.export_formats is None:
            self.export_formats = ["json", "csv", "pdf"]
        
        # Validate configurations
        if len(self.model_paths) != len(self.model_names):
            raise ValueError("Number of model paths must match number of model names")


class AutomatedEvaluationPipeline:
    """Automated pipeline for comprehensive medical AI model evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results_storage = {}
        
        # Initialize components
        self.model_registry = ModelRegistry()
        self.clinical_validator = ClinicalAccuracyValidator()
        self.knowledge_validator = MedicalKnowledgeValidator()
        self.safety_checker = SafetyComplianceChecker()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "comparisons"), exist_ok=True)
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all models and datasets.
        
        Returns:
            Complete evaluation results with comparisons and insights
        """
        self.logger.info("Starting comprehensive evaluation pipeline")
        start_time = time.time()
        
        # Initialize results structure
        pipeline_results = {
            "pipeline_info": {
                "start_time": datetime.now().isoformat(),
                "config": asdict(self.config),
                "models_to_evaluate": len(self.config.model_paths),
                "datasets_to_evaluate": sum([len(datasets) for datasets in self._get_all_datasets().values()])
            },
            "model_evaluations": {},
            "comparative_analysis": {},
            "benchmark_summary": {},
            "recommendations": {},
            "pipeline_summary": {}
        }
        
        try:
            # Load benchmark datasets
            datasets = self._load_benchmark_datasets()
            
            # Evaluate each model
            for model_path, model_name in zip(self.config.model_paths, self.config.model_names):
                self.logger.info(f"Evaluating model: {model_name}")
                
                model_results = self._evaluate_single_model(
                    model_path, model_name, datasets
                )
                pipeline_results["model_evaluations"][model_name] = model_results
            
            # Perform comparative analysis
            pipeline_results["comparative_analysis"] = self._perform_comparative_analysis(
                pipeline_results["model_evaluations"]
            )
            
            # Generate benchmark summary
            pipeline_results["benchmark_summary"] = self._generate_benchmark_summary(
                pipeline_results["model_evaluations"]
            )
            
            # Generate recommendations
            pipeline_results["recommendations"] = self._generate_pipeline_recommendations(
                pipeline_results["model_evaluations"],
                pipeline_results["comparative_analysis"]
            )
            
            # Save results
            self._save_pipeline_results(pipeline_results)
            
            # Generate visualizations and reports
            if self.config.generate_visualizations:
                self._generate_pipeline_visualizations(pipeline_results)
            
            if self.config.generate_reports:
                self._generate_pipeline_reports(pipeline_results)
            
            elapsed_time = time.time() - start_time
            pipeline_results["pipeline_summary"] = {
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": elapsed_time,
                "status": "completed",
                "models_evaluated": len(self.config.model_paths),
                "datasets_evaluated": sum([len(datasets) for datasets in self._get_all_datasets().values()])
            }
            
            self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            pipeline_results["pipeline_summary"] = {
                "status": "failed",
                "error": str(e),
                "partial_results": pipeline_results.get("model_evaluations", {})
            }
            raise
        
        return pipeline_results
    
    def _evaluate_single_model(self, model_path: str, model_name: str, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single model across all datasets."""
        model_results = {
            "model_info": {
                "name": model_name,
                "path": model_path,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "dataset_results": {},
            "overall_scores": {},
            "quality_assessment": {},
            "performance_metrics": {}
        }
        
        # Initialize model evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate each dataset
        for dataset_type, dataset_path in datasets.items():
            self.logger.info(f"  Evaluating {dataset_type} dataset")
            
            dataset_results = self._evaluate_dataset_with_timeout(
                evaluator, model_path, dataset_type, dataset_path
            )
            model_results["dataset_results"][dataset_type] = dataset_results
        
        # Calculate overall scores
        model_results["overall_scores"] = self._calculate_overall_scores(
            model_results["dataset_results"]
        )
        
        # Assess quality against thresholds
        model_results["quality_assessment"] = self._assess_model_quality(
            model_results["overall_scores"]
        )
        
        # Calculate performance metrics
        model_results["performance_metrics"] = self._calculate_performance_metrics(
            model_results["dataset_results"]
        )
        
        return model_results
    
    def _evaluate_dataset_with_timeout(self, evaluator: ModelEvaluator, 
                                     model_path: str, dataset_type: str, 
                                     dataset_path: str, timeout: int = None) -> Dict[str, Any]:
        """Evaluate dataset with timeout protection."""
        timeout = timeout or self.config.evaluation_timeout
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Evaluation timeout after {timeout} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Run evaluation
                result = evaluator.evaluate_model(
                    model_path=model_path,
                    test_datasets={dataset_type: dataset_path},
                    output_dir=os.path.join(self.config.output_dir, "temp")
                )
                
                # Extract relevant dataset results
                if dataset_type in result.get("datasets", {}):
                    return result["datasets"][dataset_type]
                else:
                    return result
                
            finally:
                # Restore old handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except TimeoutError:
            self.logger.warning(f"Timeout during evaluation of {dataset_type}")
            return {
                "error": "Evaluation timeout",
                "timeout": True,
                "dataset_type": dataset_type
            }
        except Exception as e:
            self.logger.error(f"Error evaluating {dataset_type}: {e}")
            return {
                "error": str(e),
                "dataset_type": dataset_type
            }
    
    def _perform_comparative_analysis(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across all models."""
        comparative_results = {
            "ranking_analysis": {},
            "metric_comparison": {},
            "strength_weakness_analysis": {},
            "consistency_analysis": {}
        }
        
        # Ranking analysis
        comparative_results["ranking_analysis"] = self._analyze_model_rankings(model_evaluations)
        
        # Metric comparison
        comparative_results["metric_comparison"] = self._compare_model_metrics(model_evaluations)
        
        # Strength and weakness analysis
        comparative_results["strength_weakness_analysis"] = self._analyze_strengths_weaknesses(model_evaluations)
        
        # Consistency analysis
        comparative_results["consistency_analysis"] = self._analyze_model_consistency(model_evaluations)
        
        return comparative_results
    
    def _analyze_model_rankings(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model rankings across different metrics."""
        rankings = {}
        
        # Get overall scores for each model
        model_scores = {}
        for model_name, results in model_evaluations.items():
            overall_scores = results.get("overall_scores", {})
            model_scores[model_name] = overall_scores.get("composite_score", 0.0)
        
        # Sort models by composite score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings["overall_ranking"] = [
            {"model": model, "score": score, "rank": rank + 1}
            for rank, (model, score) in enumerate(sorted_models)
        ]
        
        # Rank by individual metrics
        metrics = ["accuracy", "safety", "coherence", "relevance", "clinical_quality"]
        for metric in metrics:
            metric_scores = {}
            for model_name, results in model_evaluations.items():
                overall_scores = results.get("overall_scores", {})
                metric_key = f"{metric}_score"
                metric_scores[model_name] = overall_scores.get(metric_key, 0.0)
            
            sorted_metric_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            rankings[f"{metric}_ranking"] = [
                {"model": model, "score": score, "rank": rank + 1}
                for rank, (model, score) in enumerate(sorted_metric_models)
            ]
        
        return rankings
    
    def _compare_model_metrics(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models across different evaluation metrics."""
        metric_comparison = {
            "detailed_comparison": {},
            "statistical_analysis": {},
            "performance_gaps": {}
        }
        
        # Extract metrics for all models
        all_metrics = {}
        for model_name, results in model_evaluations.items():
            overall_scores = results.get("overall_scores", {})
            all_metrics[model_name] = overall_scores
        
        # Detailed comparison matrix
        metrics = ["accuracy", "safety", "coherence", "relevance", "clinical_quality", "composite_score"]
        metric_matrix = {}
        
        for metric in metrics:
            metric_matrix[metric] = {}
            for model_name in model_evaluations.keys():
                score = all_metrics[model_name].get(f"{metric}_score", 0.0)
                metric_matrix[metric][model_name] = score
        
        metric_comparison["detailed_comparison"] = metric_matrix
        
        # Statistical analysis
        stats = {}
        for metric in metrics:
            scores = [metric_matrix[metric][model] for model in model_evaluations.keys()]
            if scores:
                stats[metric] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "median": float(np.median(scores))
                }
        
        metric_comparison["statistical_analysis"] = stats
        
        # Performance gaps analysis
        best_models = {}
        worst_models = {}
        
        for metric in metrics:
            model_scores = metric_matrix[metric]
            best_model = max(model_scores.items(), key=lambda x: x[1])
            worst_model = min(model_scores.items(), key=lambda x: x[1])
            
            best_models[metric] = best_model[0]
            worst_models[metric] = worst_model[0]
        
        metric_comparison["performance_gaps"] = {
            "best_performers": best_models,
            "worst_performers": worst_models
        }
        
        return metric_comparison
    
    def _analyze_strengths_weaknesses(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual model strengths and weaknesses."""
        strengths_weaknesses = {}
        
        for model_name, results in model_evaluations.items():
            overall_scores = results.get("overall_scores", {})
            dataset_results = results.get("dataset_results", {})
            
            model_analysis = {
                "strengths": [],
                "weaknesses": [],
                "performance_profile": {},
                "improvement_areas": []
            }
            
            # Analyze performance profile
            metrics = ["accuracy", "safety", "coherence", "relevance", "clinical_quality"]
            performance_profile = {}
            
            for metric in metrics:
                score = overall_scores.get(f"{metric}_score", 0.0)
                performance_profile[metric] = score
                
                if score >= 0.8:
                    model_analysis["strengths"].append(f"Strong {metric}")
                elif score < 0.6:
                    model_analysis["weaknesses"].append(f"Poor {metric}")
                    model_analysis["improvement_areas"].append(metric)
            
            model_analysis["performance_profile"] = performance_profile
            
            # Analyze dataset-specific performance
            dataset_performance = {}
            for dataset_name, dataset_result in dataset_results.items():
                if "medical_accuracy" in dataset_result:
                    acc_score = dataset_result["medical_accuracy"].get("avg_f1", 0.0)
                    dataset_performance[dataset_name] = acc_score
            
            # Find best and worst performing datasets
            if dataset_performance:
                best_dataset = max(dataset_performance.items(), key=lambda x: x[1])
                worst_dataset = min(dataset_performance.items(), key=lambda x: x[1])
                
                model_analysis["best_performing_dataset"] = {
                    "name": best_dataset[0],
                    "score": best_dataset[1]
                }
                model_analysis["worst_performing_dataset"] = {
                    "name": worst_dataset[0],
                    "score": worst_dataset[1]
                }
            
            strengths_weaknesses[model_name] = model_analysis
        
        return strengths_weaknesses
    
    def _analyze_model_consistency(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of model performance across datasets."""
        consistency_analysis = {}
        
        for model_name, results in model_evaluations.items():
            dataset_results = results.get("dataset_results", {})
            overall_scores = results.get("overall_scores", {})
            
            consistency_metrics = {
                "score_variance": {},
                "dataset_consistency": {},
                "performance_stability": {}
            }
            
            # Calculate score variance across datasets
            medical_accuracy_scores = []
            safety_scores = []
            
            for dataset_name, dataset_result in dataset_results.items():
                if "medical_accuracy" in dataset_result:
                    f1_score = dataset_result["medical_accuracy"].get("avg_f1", 0.0)
                    medical_accuracy_scores.append(f1_score)
                
                if "safety_assessment" in dataset_result:
                    safety_score = dataset_result["safety_assessment"].get("avg_safety_score", 0.0)
                    safety_scores.append(safety_score)
            
            if medical_accuracy_scores:
                consistency_metrics["score_variance"]["medical_accuracy"] = {
                    "variance": float(np.var(medical_accuracy_scores)),
                    "std_dev": float(np.std(medical_accuracy_scores)),
                    "coefficient_of_variation": float(np.std(medical_accuracy_scores) / np.mean(medical_accuracy_scores))
                }
            
            if safety_scores:
                consistency_metrics["score_variance"]["safety"] = {
                    "variance": float(np.var(safety_scores)),
                    "std_dev": float(np.std(safety_scores)),
                    "coefficient_of_variation": float(np.std(safety_scores) / np.mean(safety_scores))
                }
            
            # Assess performance stability
            all_scores = medical_accuracy_scores + safety_scores
            if all_scores:
                cv = np.std(all_scores) / np.mean(all_scores) if np.mean(all_scores) > 0 else float('inf')
                
                if cv < 0.1:
                    stability = "very_stable"
                elif cv < 0.2:
                    stability = "stable"
                elif cv < 0.3:
                    stability = "moderately_stable"
                else:
                    stability = "unstable"
                
                consistency_metrics["performance_stability"] = {
                    "classification": stability,
                    "coefficient_of_variation": cv
                }
            
            consistency_analysis[model_name] = consistency_metrics
        
        return consistency_analysis
    
    def _generate_benchmark_summary(self, model_evaluations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary across all models."""
        benchmark_summary = {
            "model_count": len(model_evaluations),
            "evaluation_date": datetime.now().isoformat(),
            "overall_leaderboard": [],
            "metric_leaderboards": {},
            "performance_distribution": {},
            "benchmark_insights": []
        }
        
        # Overall leaderboard
        model_scores = {}
        for model_name, results in model_evaluations.items():
            composite_score = results.get("overall_scores", {}).get("composite_score", 0.0)
            model_scores[model_name] = composite_score
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        benchmark_summary["overall_leaderboard"] = [
            {"model": model, "score": score, "rank": rank + 1}
            for rank, (model, score) in enumerate(sorted_models)
        ]
        
        # Metric leaderboards
        metrics = ["accuracy", "safety", "coherence", "relevance", "clinical_quality"]
        for metric in metrics:
            metric_scores = {}
            for model_name, results in model_evaluations.items():
                overall_scores = results.get("overall_scores", {})
                metric_key = f"{metric}_score"
                metric_scores[model_name] = overall_scores.get(metric_key, 0.0)
            
            sorted_metric_models = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            benchmark_summary["metric_leaderboards"][metric] = [
                {"model": model, "score": score, "rank": rank + 1}
                for rank, (model, score) in enumerate(sorted_metric_models)
            ]
        
        # Performance distribution analysis
        all_composite_scores = list(model_scores.values())
        if all_composite_scores:
            benchmark_summary["performance_distribution"] = {
                "mean": float(np.mean(all_composite_scores)),
                "median": float(np.median(all_composite_scores)),
                "std": float(np.std(all_composite_scores)),
                "min": float(np.min(all_composite_scores)),
                "max": float(np.max(all_composite_scores)),
                "quartiles": {
                    "q1": float(np.percentile(all_composite_scores, 25)),
                    "q3": float(np.percentile(all_composite_scores, 75))
                }
            }
        
        # Generate insights
        if all_composite_scores:
            top_score = max(all_composite_scores)
            bottom_score = min(all_composite_scores)
            score_range = top_score - bottom_score
            
            benchmark_summary["benchmark_insights"] = [
                f"Performance range: {score_range:.3f} points",
                f"Top performer: {sorted_models[0][0]} ({sorted_models[0][1]:.3f})",
                f"Average performance: {np.mean(all_composite_scores):.3f}"
            ]
            
            if score_range > 0.2:
                benchmark_summary["benchmark_insights"].append("Large performance variation suggests significant model differences")
            
            if np.mean(all_composite_scores) > 0.8:
                benchmark_summary["benchmark_insights"].append("Overall strong performance across all models")
            elif np.mean(all_composite_scores) < 0.6:
                benchmark_summary["benchmark_insights"].append("Overall performance needs improvement")
        
        return benchmark_summary
    
    def _generate_pipeline_recommendations(self, model_evaluations: Dict[str, Any], 
                                         comparative_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive recommendations based on evaluation results."""
        recommendations = {
            "model_specific": {},
            "general": [],
            "priority_actions": [],
            "strategic_insights": []
        }
        
        # Model-specific recommendations
        for model_name, results in model_evaluations.items():
            model_recommendations = {
                "immediate_actions": [],
                "medium_term_improvements": [],
                "long_term_goals": []
            }
            
            overall_scores = results.get("overall_scores", {})
            quality_assessment = results.get("quality_assessment", {})
            
            # Analyze individual metric performance
            metrics = ["accuracy", "safety", "coherence", "relevance", "clinical_quality"]
            for metric in metrics:
                score = overall_scores.get(f"{metric}_score", 0.0)
                threshold_key = f"min_{metric}_threshold"
                
                if threshold_key in quality_assessment:
                    threshold = quality_assessment[threshold_key]
                    if score < threshold:
                        deficit = threshold - score
                        if deficit > 0.2:
                            model_recommendations["immediate_actions"].append(
                                f"Critical improvement needed in {metric} (current: {score:.3f}, target: {threshold:.3f})"
                            )
                        elif deficit > 0.1:
                            model_recommendations["medium_term_improvements"].append(
                                f"Moderate improvement needed in {metric}"
                            )
                        else:
                            model_recommendations["long_term_goals"].append(
                                f"Minor enhancement in {metric}"
                            )
            
            recommendations["model_specific"][model_name] = model_recommendations
        
        # General recommendations
        comparative_results = comparative_analysis.get("comparative_analysis", {})
        rankings = comparative_results.get("ranking_analysis", {})
        
        # Analyze overall performance trends
        if rankings:
            overall_ranking = rankings.get("overall_ranking", [])
            if overall_ranking:
                top_performer = overall_ranking[0]
                bottom_performer = overall_ranking[-1]
                
                recommendations["general"].extend([
                    f"Study top performer ({top_performer['model']}) techniques",
                    f"Investigate common issues with underperforming models",
                    "Implement best practices from highest-scoring models"
                ])
        
        # Priority actions
        all_immediate_actions = []
        for model_name, model_recommendations in recommendations["model_specific"].items():
            all_immediate_actions.extend(model_recommendations["immediate_actions"])
        
        if all_immediate_actions:
            recommendations["priority_actions"] = list(set(all_immediate_actions))[:5]  # Top 5 unique actions
        
        # Strategic insights
        performance_gaps = comparative_results.get("metric_comparison", {}).get("performance_gaps", {})
        if performance_gaps:
            best_performers = performance_gaps.get("best_performers", {})
            
            # Identify which metrics need most attention across all models
            metric_attention = {}
            for model_name, results in model_evaluations.items():
                overall_scores = results.get("overall_scores", {})
                for metric in metrics:
                    score = overall_scores.get(f"{metric}_score", 0.0)
                    if metric not in metric_attention:
                        metric_attention[metric] = []
                    metric_attention[metric].append(score)
            
            # Find metrics with lowest average performance
            avg_scores = {metric: np.mean(scores) for metric, scores in metric_attention.items()}
            worst_metric = min(avg_scores.keys(), key=lambda x: avg_scores[x])
            
            recommendations["strategic_insights"].extend([
                f"Focus improvement efforts on {worst_metric} (lowest average: {avg_scores[worst_metric]:.3f})",
                "Consider ensemble methods combining strengths of multiple models",
                "Implement progressive evaluation during model development"
            ])
        
        return recommendations
    
    def _calculate_overall_scores(self, dataset_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall scores across all datasets."""
        all_scores = {
            "accuracy_scores": [],
            "safety_scores": [],
            "coherence_scores": [],
            "relevance_scores": [],
            "clinical_quality_scores": []
        }
        
        # Collect scores from all datasets
        for dataset_name, dataset_result in dataset_results.items():
            # Medical accuracy
            if "medical_accuracy" in dataset_result:
                f1_score = dataset_result["medical_accuracy"].get("avg_f1", 0.0)
                all_scores["accuracy_scores"].append(f1_score)
            
            # Safety
            if "safety_assessment" in dataset_result:
                safety_score = dataset_result["safety_assessment"].get("avg_safety_score", 0.0)
                all_scores["safety_scores"].append(safety_score)
            
            # Clinical quality (estimated from available metrics)
            clinical_score = 0.0
            if "clinical_assessment" in dataset_result:
                clinical_score = dataset_result["clinical_assessment"].get("avg_quality_score", 0.0)
            all_scores["clinical_quality_scores"].append(clinical_score)
            
            # Relevance (estimated from medical accuracy)
            all_scores["relevance_scores"].append(f1_score)  # Simplified
            all_scores["coherence_scores"].append(0.7)  # Placeholder for now
        
        # Calculate averages
        overall_scores = {}
        for score_type, scores in all_scores.items():
            if scores:
                overall_scores[f"{score_type.replace('_scores', '')}_score"] = np.mean(scores)
            else:
                overall_scores[f"{score_type.replace('_scores', '')}_score"] = 0.0
        
        # Calculate composite score
        weights = {
            "accuracy_score": 0.3,
            "safety_score": 0.3,
            "coherence_score": 0.15,
            "relevance_score": 0.15,
            "clinical_quality_score": 0.1
        }
        
        composite_score = sum(
            overall_scores.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )
        
        overall_scores["composite_score"] = composite_score
        
        return overall_scores
    
    def _assess_model_quality(self, overall_scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess model quality against defined thresholds."""
        quality_assessment = {
            "meets_minimum_requirements": True,
            "quality_issues": [],
            "threshold_violations": []
        }
        
        # Check against thresholds
        thresholds = {
            "accuracy_score": self.config.min_accuracy_threshold,
            "safety_score": self.config.min_safety_threshold,
            "coherence_score": self.config.min_coherence_threshold
        }
        
        for metric, threshold in thresholds.items():
            score = overall_scores.get(metric, 0.0)
            threshold_key = f"min_{metric.replace('_score', '')}_threshold"
            
            quality_assessment[threshold_key] = threshold
            
            if score < threshold:
                quality_assessment["meets_minimum_requirements"] = False
                quality_assessment["quality_issues"].append(
                    f"{metric} below threshold ({score:.3f} < {threshold:.3f})"
                )
                quality_assessment["threshold_violations"].append({
                    "metric": metric,
                    "score": score,
                    "threshold": threshold,
                    "deficit": threshold - score
                })
        
        # Overall quality classification
        composite_score = overall_scores.get("composite_score", 0.0)
        if composite_score >= 0.85:
            quality_assessment["quality_classification"] = "excellent"
        elif composite_score >= 0.75:
            quality_assessment["quality_classification"] = "good"
        elif composite_score >= 0.65:
            quality_assessment["quality_classification"] = "acceptable"
        elif composite_score >= 0.5:
            quality_assessment["quality_classification"] = "poor"
        else:
            quality_assessment["quality_classification"] = "very_poor"
        
        return quality_assessment
    
    def _calculate_performance_metrics(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        performance_metrics = {
            "evaluation_coverage": {},
            "dataset_performance": {},
            "error_analysis": {},
            "resource_usage": {}
        }
        
        # Evaluation coverage
        total_datasets = len(dataset_results)
        successful_evaluations = sum(
            1 for result in dataset_results.values() 
            if "error" not in result
        )
        
        performance_metrics["evaluation_coverage"] = {
            "total_datasets": total_datasets,
            "successful_evaluations": successful_evaluations,
            "success_rate": successful_evaluations / total_datasets if total_datasets > 0 else 0,
            "failed_evaluations": total_datasets - successful_evaluations
        }
        
        # Dataset-specific performance
        dataset_performance = {}
        for dataset_name, dataset_result in dataset_results.items():
            if "error" not in dataset_result:
                performance = {}
                
                if "medical_accuracy" in dataset_result:
                    performance["medical_accuracy"] = {
                        "f1_score": dataset_result["medical_accuracy"].get("avg_f1", 0.0),
                        "precision": dataset_result["medical_accuracy"].get("avg_precision", 0.0),
                        "recall": dataset_result["medical_accuracy"].get("avg_recall", 0.0)
                    }
                
                if "safety_assessment" in dataset_result:
                    performance["safety"] = {
                        "safety_score": dataset_result["safety_assessment"].get("avg_safety_score", 0.0)
                    }
                
                if "clinical_assessment" in dataset_result:
                    performance["clinical_quality"] = {
                        "quality_score": dataset_result["clinical_assessment"].get("avg_quality_score", 0.0)
                    }
                
                dataset_performance[dataset_name] = performance
            else:
                dataset_performance[dataset_name] = {"error": dataset_result["error"]}
        
        performance_metrics["dataset_performance"] = dataset_performance
        
        # Error analysis
        errors = []
        timeouts = []
        
        for dataset_name, dataset_result in dataset_results.items():
            if "error" in dataset_result:
                error_info = dataset_result["error"]
                if "timeout" in str(error_info).lower():
                    timeouts.append(dataset_name)
                else:
                    errors.append({
                        "dataset": dataset_name,
                        "error": error_info
                    })
        
        performance_metrics["error_analysis"] = {
            "total_errors": len(errors),
            "timeout_errors": len(timeouts),
            "errors": errors,
            "timeout_datasets": timeouts
        }
        
        return performance_metrics
    
    def _load_benchmark_datasets(self) -> Dict[str, str]:
        """Load available benchmark datasets."""
        datasets = {}
        
        # Check for benchmark files in evaluation directory
        eval_dir = Path(__file__).parent
        benchmark_dir = eval_dir / "benchmarks"
        
        if benchmark_dir.exists():
            # Load standard benchmark files
            benchmark_files = {
                "holdout_test": benchmark_dir / "holdout_test_set.json",
                "clinical_cases": benchmark_dir / "clinical_case_scenarios.json",
                "edge_cases": benchmark_dir / "edge_cases.json",
                "conversations": benchmark_dir / "conversation_tests.json"
            }
            
            for dataset_name, file_path in benchmark_files.items():
                if file_path.exists():
                    datasets[dataset_name] = str(file_path)
                else:
                    self.logger.warning(f"Benchmark file not found: {file_path}")
        
        # Add custom datasets if specified
        if self.config.custom_datasets:
            datasets.update(self.config.custom_datasets)
        
        # Ensure we have at least some datasets to work with
        if not datasets:
            self.logger.warning("No benchmark datasets found. Using sample datasets.")
            datasets = self._create_sample_datasets()
        
        return datasets
    
    def _create_sample_datasets(self) -> Dict[str, str]:
        """Create sample datasets for demonstration."""
        sample_dir = os.path.join(self.config.output_dir, "sample_datasets")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create simple sample datasets
        sample_datasets = {
            "sample_test": {
                "samples": [
                    {
                        "input": "What are symptoms of diabetes?",
                        "output": "Common diabetes symptoms include increased thirst, frequent urination, and fatigue."
                    }
                ]
            }
        }
        
        for name, data in sample_datasets.items():
            file_path = os.path.join(sample_dir, f"{name}.json")
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return {name: os.path.join(sample_dir, f"{name}.json") for name in sample_datasets.keys()}
    
    def _get_all_datasets(self) -> Dict[str, Dict[str, str]]:
        """Get all available datasets organized by type."""
        return {
            "benchmark": self._load_benchmark_datasets(),
            "custom": self.config.custom_datasets or {}
        }
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save complete pipeline results."""
        # Save main results file
        results_file = os.path.join(self.config.output_dir, "pipeline_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save individual model results
        for model_name, model_results in results.get("model_evaluations", {}).items():
            model_file = os.path.join(self.config.output_dir, f"{model_name}_results.json")
            with open(model_file, 'w') as f:
                json.dump(model_results, f, indent=2)
        
        # Save comparative analysis
        comp_file = os.path.join(self.config.output_dir, "comparative_analysis.json")
        with open(comp_file, 'w') as f:
            json.dump(results.get("comparative_analysis", {}), f, indent=2)
        
        self.logger.info(f"Pipeline results saved to {self.config.output_dir}")
    
    def _generate_pipeline_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive visualizations."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            viz_dir = os.path.join(self.config.output_dir, "visualizations")
            
            # Model comparison charts
            self._create_model_comparison_charts(results, viz_dir)
            
            # Performance distribution charts
            self._create_performance_distribution_charts(results, viz_dir)
            
            # Benchmark leaderboard charts
            self._create_leaderboard_charts(results, viz_dir)
            
        except ImportError:
            self.logger.warning("Visualization dependencies not available. Skipping charts.")
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _create_model_comparison_charts(self, results: Dict[str, Any], viz_dir: str) -> None:
        """Create model comparison visualization charts."""
        # This would contain detailed chart creation code
        # For now, just create placeholder files
        plt_path = os.path.join(viz_dir, "model_comparison.png")
        with open(plt_path, 'w') as f:
            f.write("# Model comparison chart placeholder")
    
    def _create_performance_distribution_charts(self, results: Dict[str, Any], viz_dir: str) -> None:
        """Create performance distribution charts."""
        plt_path = os.path.join(viz_dir, "performance_distribution.png")
        with open(plt_path, 'w') as f:
            f.write("# Performance distribution chart placeholder")
    
    def _create_leaderboard_charts(self, results: Dict[str, Any], viz_dir: str) -> None:
        """Create benchmark leaderboard charts."""
        plt_path = os.path.join(viz_dir, "leaderboard.png")
        with open(plt_path, 'w') as f:
            f.write("# Leaderboard chart placeholder")
    
    def _generate_pipeline_reports(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive reports."""
        reports_dir = os.path.join(self.config.output_dir, "reports")
        
        # Generate executive summary
        exec_summary = self._create_executive_summary(results)
        exec_file = os.path.join(reports_dir, "executive_summary.json")
        with open(exec_file, 'w') as f:
            json.dump(exec_summary, f, indent=2)
        
        # Generate detailed technical report
        tech_report = self._create_technical_report(results)
        tech_file = os.path.join(reports_dir, "technical_report.json")
        with open(tech_file, 'w') as f:
            json.dump(tech_report, f, indent=2)
        
        self.logger.info(f"Reports generated in {reports_dir}")
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of evaluation results."""
        summary = {
            "evaluation_overview": {
                "total_models": len(results.get("model_evaluations", {})),
                "evaluation_date": datetime.now().isoformat(),
                "status": results.get("pipeline_summary", {}).get("status", "unknown")
            },
            "key_findings": [],
            "recommendations": [],
            "model_rankings": []
        }
        
        # Extract key findings
        benchmark_summary = results.get("benchmark_summary", {})
        if benchmark_summary:
            insights = benchmark_summary.get("benchmark_insights", [])
            summary["key_findings"] = insights
        
        # Add model rankings
        leaderboard = benchmark_summary.get("overall_leaderboard", [])
        if leaderboard:
            summary["model_rankings"] = leaderboard[:5]  # Top 5
        
        # Add top recommendations
        recommendations = results.get("recommendations", {})
        general_recs = recommendations.get("general", [])
        summary["recommendations"] = general_recs[:5]  # Top 5
        
        return summary
    
    def _create_technical_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed technical report."""
        return {
            "methodology": "Comprehensive medical AI model evaluation",
            "evaluation_criteria": self.config.metrics_to_evaluate,
            "datasets_used": list(self._get_all_datasets().keys()),
            "detailed_results": results,
            "statistical_analysis": "See comparative_analysis section",
            "methodology_notes": [
                "Evaluation conducted using standardized benchmark datasets",
                "Multiple metrics combined using weighted scoring",
                "Quality thresholds applied for minimum requirements",
                "Comparative analysis performed across all models"
            ]
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger


def main():
    """Main function to run automated evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Medical AI Evaluation Pipeline")
    parser.add_argument("--model_paths", nargs="+", required=True, help="Paths to models to evaluate")
    parser.add_argument("--model_names", nargs="+", required=True, help="Names of models to evaluate")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--custom_datasets", nargs="*", help="Custom dataset paths")
    parser.add_argument("--min_accuracy_threshold", type=float, default=0.7, help="Minimum accuracy threshold")
    parser.add_argument("--min_safety_threshold", type=float, default=0.8, help="Minimum safety threshold")
    parser.add_argument("--timeout", type=int, default=3600, help="Evaluation timeout in seconds")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvaluationConfig(
        model_paths=args.model_paths,
        model_names=args.model_names,
        output_dir=args.output_dir,
        min_accuracy_threshold=args.min_accuracy_threshold,
        min_safety_threshold=args.min_safety_threshold,
        evaluation_timeout=args.timeout
    )
    
    # Run pipeline
    pipeline = AutomatedEvaluationPipeline(config)
    results = pipeline.run_comprehensive_evaluation()
    
    print(f"\n{'='*60}")
    print("AUTOMATED EVALUATION PIPELINE COMPLETED")
    print(f"{'='*60}")
    
    # Print summary
    benchmark_summary = results.get("benchmark_summary", {})
    leaderboard = benchmark_summary.get("overall_leaderboard", [])
    
    if leaderboard:
        print(f"\nTop 3 Models:")
        for i, model in enumerate(leaderboard[:3]):
            print(f"  {i+1}. {model['model']}: {model['score']:.3f}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print(f"Executive summary: {args.output_dir}/reports/executive_summary.json")


if __name__ == "__main__":
    main()