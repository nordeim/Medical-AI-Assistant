#!/usr/bin/env python3
"""
Comprehensive Medical AI Model Evaluation Script

This script provides a complete evaluation framework for medical AI models, including:
- Medical accuracy metrics (precision, recall, F1)
- Clinical assessment quality scores
- Conversation coherence evaluation
- Safety and appropriateness checks
- Response relevance scoring

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

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
    SafetyComplianceChecker,
    ExpertReviewIntegrator
)
from utils.data_validator import DataValidator
from utils.model_registry import ModelRegistry


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for medical AI models.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            config_path: Path to evaluation configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize evaluation components
        self.medical_accuracy = MedicalAccuracyMetrics()
        self.clinical_assessment = ClinicalAssessmentMetrics()
        self.conversation_coherence = ConversationCoherenceMetrics()
        self.safety_assessment = SafetyAssessmentMetrics()
        self.relevance_scoring = RelevanceScoringMetrics()
        
        # Initialize clinical validators
        self.clinical_validator = ClinicalAccuracyValidator()
        self.knowledge_validator = MedicalKnowledgeValidator()
        self.safety_checker = SafetyComplianceChecker()
        self.expert_integrator = ExpertReviewIntegrator()
        
        # Data validator for PHI protection
        self.data_validator = DataValidator()
        
        self.evaluation_results = {}
        self.start_time = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
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
    
    def _load_config(self, config_path: str) -> Dict:
        """Load evaluation configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default evaluation configuration."""
        return {
            "evaluation_datasets": {
                "test_sets": {
                    "holdout_test": {
                        "path": "data/test/holdout/",
                        "type": "standard",
                        "size": 1000
                    },
                    "clinical_cases": {
                        "path": "data/test/clinical_cases/",
                        "type": "clinical_scenarios",
                        "size": 500
                    },
                    "edge_cases": {
                        "path": "data/test/edge_cases/",
                        "type": "edge_case_validation",
                        "size": 200
                    }
                },
                "conversation_tests": {
                    "multi_turn": {
                        "path": "data/test/conversations/",
                        "type": "multi_turn_conversation",
                        "max_turns": 10
                    }
                }
            },
            "metrics": {
                "medical_accuracy": {
                    "enabled": True,
                    "weights": {"precision": 0.33, "recall": 0.33, "f1": 0.34}
                },
                "clinical_assessment": {
                    "enabled": True,
                    "quality_threshold": 0.8
                },
                "conversation_coherence": {
                    "enabled": True,
                    "coherence_threshold": 0.7
                },
                "safety_assessment": {
                    "enabled": True,
                    "safety_threshold": 0.9
                },
                "relevance_scoring": {
                    "enabled": True,
                    "relevance_threshold": 0.75
                }
            },
            "visualization": {
                "enabled": True,
                "output_dir": "evaluation_results/visualizations/",
                "format": ["png", "pdf"]
            },
            "reports": {
                "detailed": True,
                "summary": True,
                "benchmark_comparison": True
            }
        }
    
    def evaluate_model(self, 
                      model_path: str,
                      test_datasets: Optional[List[str]] = None,
                      output_dir: str = "evaluation_results") -> Dict:
        """
        Perform comprehensive evaluation of a medical AI model.
        
        Args:
            model_path: Path to the model to evaluate
            test_datasets: List of test dataset paths
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        self.logger.info(f"Starting comprehensive model evaluation: {model_path}")
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Load model and tokenizer
        model, tokenizer = self._load_model(model_path)
        
        # Initialize evaluation datasets
        if test_datasets is None:
            test_datasets = self._load_default_datasets()
        
        # Perform evaluations
        results = {
            "model_info": self._get_model_info(model_path),
            "evaluation_timestamp": datetime.now().isoformat(),
            "datasets": {},
            "metrics": {},
            "clinical_validation": {},
            "safety_assessment": {},
            "performance_summary": {}
        }
        
        # Evaluate each dataset
        for dataset_name, dataset_path in test_datasets.items():
            self.logger.info(f"Evaluating dataset: {dataset_name}")
            dataset_results = self._evaluate_dataset(
                model, tokenizer, dataset_name, dataset_path
            )
            results["datasets"][dataset_name] = dataset_results
        
        # Aggregate results
        results["metrics"] = self._aggregate_metrics(results["datasets"])
        results["clinical_validation"] = self._run_clinical_validation(results["datasets"])
        results["safety_assessment"] = self._run_safety_assessment(results["datasets"])
        
        # Generate comprehensive reports
        results["performance_summary"] = self._generate_performance_summary(results)
        
        # Save results
        self._save_results(results, output_dir)
        
        # Generate visualizations
        if self.config["visualization"]["enabled"]:
            self._generate_visualizations(results, output_dir)
        
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _load_model(self, model_path: str) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _load_default_datasets(self) -> Dict[str, str]:
        """Load default evaluation datasets."""
        datasets = {}
        
        # Check for datasets in the config
        for dataset_name, dataset_config in self.config["evaluation_datasets"]["test_sets"].items():
            dataset_path = dataset_config["path"]
            if os.path.exists(dataset_path):
                datasets[dataset_name] = dataset_path
        
        # Check for conversation datasets
        for conv_name, conv_config in self.config["evaluation_datasets"]["conversation_tests"].items():
            conv_path = conv_config["path"]
            if os.path.exists(conv_path):
                datasets[conv_name] = conv_path
        
        return datasets
    
    def _evaluate_dataset(self, 
                         model: Any, 
                         tokenizer: Any, 
                         dataset_name: str, 
                         dataset_path: str) -> Dict:
        """Evaluate model on a specific dataset."""
        dataset_results = {
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "samples_evaluated": 0,
            "medical_accuracy": {},
            "clinical_assessment": {},
            "conversation_coherence": {},
            "safety_assessment": {},
            "relevance_scoring": {},
            "error_analysis": {},
            "sample_results": []
        }
        
        try:
            # Load dataset
            dataset = self._load_dataset(dataset_path)
            dataset_results["samples_evaluated"] = len(dataset)
            
            # Evaluate each sample
            for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
                sample_result = self._evaluate_sample(model, tokenizer, sample)
                dataset_results["sample_results"].append(sample_result)
            
            # Aggregate sample results
            self._aggregate_sample_results(dataset_results)
            
        except Exception as e:
            self.logger.error(f"Error evaluating dataset {dataset_name}: {e}")
            dataset_results["error"] = str(e)
        
        return dataset_results
    
    def _load_dataset(self, dataset_path: str) -> Dataset:
        """Load dataset from path."""
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_list(data)
        elif dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path)
            return Dataset.from_pandas(data)
        else:
            # Assume it's a directory of files
            files = []
            for file in os.listdir(dataset_path):
                if file.endswith(('.json', '.csv')):
                    files.append(os.path.join(dataset_path, file))
            
            datasets = []
            for file in files:
                if file.endswith('.json'):
                    with open(file, 'r') as f:
                        data = json.load(f)
                    datasets.extend(data if isinstance(data, list) else [data])
                elif file.endswith('.csv'):
                    data = pd.read_csv(file)
                    datasets.extend(data.to_dict('records'))
            
            return Dataset.from_list(datasets)
    
    def _evaluate_sample(self, model: Any, tokenizer: Any, sample: Dict) -> Dict:
        """Evaluate model on a single sample."""
        sample_result = {
            "input": sample.get("input", ""),
            "expected_output": sample.get("output", ""),
            "model_output": "",
            "medical_accuracy": {},
            "clinical_assessment": {},
            "conversation_coherence": {},
            "safety_assessment": {},
            "relevance_scoring": {}
        }
        
        try:
            # Generate model output
            inputs = tokenizer(sample["input"], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
            
            sample_result["model_output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Evaluate medical accuracy
            if self.config["metrics"]["medical_accuracy"]["enabled"]:
                sample_result["medical_accuracy"] = self.medical_accuracy.evaluate(
                    sample_result["expected_output"], 
                    sample_result["model_output"]
                )
            
            # Evaluate clinical assessment quality
            if self.config["metrics"]["clinical_assessment"]["enabled"]:
                sample_result["clinical_assessment"] = self.clinical_assessment.evaluate(
                    sample_result["input"],
                    sample_result["model_output"]
                )
            
            # Evaluate conversation coherence (for conversation datasets)
            if "conversation" in str(sample).lower() and self.config["metrics"]["conversation_coherence"]["enabled"]:
                sample_result["conversation_coherence"] = self.conversation_coherence.evaluate(
                    sample.get("context", ""),
                    sample_result["model_output"]
                )
            
            # Evaluate safety
            if self.config["metrics"]["safety_assessment"]["enabled"]:
                sample_result["safety_assessment"] = self.safety_assessment.evaluate(
                    sample_result["model_output"]
                )
            
            # Evaluate relevance
            if self.config["metrics"]["relevance_scoring"]["enabled"]:
                sample_result["relevance_scoring"] = self.relevance_scoring.evaluate(
                    sample_result["input"],
                    sample_result["model_output"]
                )
        
        except Exception as e:
            sample_result["error"] = str(e)
            self.logger.warning(f"Error evaluating sample: {e}")
        
        return sample_result
    
    def _aggregate_sample_results(self, dataset_results: Dict) -> None:
        """Aggregate results from individual samples."""
        samples = dataset_results["sample_results"]
        
        # Aggregate medical accuracy metrics
        if samples and "medical_accuracy" in samples[0]:
            all_precisions = [s["medical_accuracy"].get("precision", 0) for s in samples if "medical_accuracy" in s]
            all_recalls = [s["medical_accuracy"].get("recall", 0) for s in samples if "medical_accuracy" in s]
            all_f1s = [s["medical_accuracy"].get("f1", 0) for s in samples if "medical_accuracy" in s]
            
            dataset_results["medical_accuracy"] = {
                "avg_precision": np.mean(all_precisions) if all_precisions else 0,
                "avg_recall": np.mean(all_recalls) if all_recalls else 0,
                "avg_f1": np.mean(all_f1s) if all_f1s else 0,
                "std_precision": np.std(all_precisions) if all_precisions else 0,
                "std_recall": np.std(all_recalls) if all_recalls else 0,
                "std_f1": np.std(all_f1s) if all_f1s else 0
            }
        
        # Aggregate clinical assessment scores
        if samples and "clinical_assessment" in samples[0]:
            all_scores = [s["clinical_assessment"].get("quality_score", 0) for s in samples if "clinical_assessment" in s]
            dataset_results["clinical_assessment"] = {
                "avg_quality_score": np.mean(all_scores) if all_scores else 0,
                "std_quality_score": np.std(all_scores) if all_scores else 0
            }
        
        # Aggregate safety scores
        if samples and "safety_assessment" in samples[0]:
            all_safety_scores = [s["safety_assessment"].get("safety_score", 0) for s in samples if "safety_assessment" in s]
            dataset_results["safety_assessment"] = {
                "avg_safety_score": np.mean(all_safety_scores) if all_safety_scores else 0,
                "std_safety_score": np.std(all_safety_scores) if all_safety_scores else 0
            }
        
        # Aggregate relevance scores
        if samples and "relevance_scoring" in samples[0]:
            all_relevance_scores = [s["relevance_scoring"].get("relevance_score", 0) for s in samples if "relevance_scoring" in s]
            dataset_results["relevance_scoring"] = {
                "avg_relevance_score": np.mean(all_relevance_scores) if all_relevance_scores else 0,
                "std_relevance_score": np.std(all_relevance_scores) if all_relevance_scores else 0
            }
    
    def _aggregate_metrics(self, datasets_results: Dict) -> Dict:
        """Aggregate metrics across all datasets."""
        aggregated_metrics = {
            "overall_performance": {},
            "dataset_comparison": {},
            "metric_distributions": {}
        }
        
        # Calculate overall performance metrics
        all_medical_metrics = []
        all_clinical_scores = []
        all_safety_scores = []
        all_relevance_scores = []
        
        for dataset_name, dataset_results in datasets_results.items():
            if "medical_accuracy" in dataset_results:
                medical = dataset_results["medical_accuracy"]
                all_medical_metrics.append({
                    "precision": medical.get("avg_precision", 0),
                    "recall": medical.get("avg_recall", 0),
                    "f1": medical.get("avg_f1", 0)
                })
            
            if "clinical_assessment" in dataset_results:
                clinical = dataset_results["clinical_assessment"]
                all_clinical_scores.append(clinical.get("avg_quality_score", 0))
            
            if "safety_assessment" in dataset_results:
                safety = dataset_results["safety_assessment"]
                all_safety_scores.append(safety.get("avg_safety_score", 0))
            
            if "relevance_scoring" in dataset_results:
                relevance = dataset_results["relevance_scoring"]
                all_relevance_scores.append(relevance.get("avg_relevance_score", 0))
        
        # Calculate overall averages
        if all_medical_metrics:
            avg_precision = np.mean([m["precision"] for m in all_medical_metrics])
            avg_recall = np.mean([m["recall"] for m in all_medical_metrics])
            avg_f1 = np.mean([m["f1"] for m in all_medical_metrics])
            
            aggregated_metrics["overall_performance"]["medical_accuracy"] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            }
        
        if all_clinical_scores:
            aggregated_metrics["overall_performance"]["clinical_assessment"] = {
                "avg_quality_score": np.mean(all_clinical_scores)
            }
        
        if all_safety_scores:
            aggregated_metrics["overall_performance"]["safety_assessment"] = {
                "avg_safety_score": np.mean(all_safety_scores)
            }
        
        if all_relevance_scores:
            aggregated_metrics["overall_performance"]["relevance_scoring"] = {
                "avg_relevance_score": np.mean(all_relevance_scores)
            }
        
        return aggregated_metrics
    
    def _run_clinical_validation(self, datasets_results: Dict) -> Dict:
        """Run clinical validation across all datasets."""
        clinical_results = {
            "accuracy_assessment": {},
            "knowledge_validation": {},
            "expert_review": {}
        }
        
        try:
            # Run clinical accuracy assessment
            for dataset_name, dataset_results in datasets_results.items():
                sample_outputs = [s.get("model_output", "") for s in dataset_results.get("sample_results", [])]
                clinical_results["accuracy_assessment"][dataset_name] = self.clinical_validator.assess_accuracy(
                    sample_outputs
                )
            
            # Run medical knowledge validation
            all_model_outputs = []
            for dataset_results in datasets_results.values():
                all_model_outputs.extend([s.get("model_output", "") for s in dataset_results.get("sample_results", [])])
            
            clinical_results["knowledge_validation"] = self.knowledge_validator.validate_knowledge(
                all_model_outputs
            )
            
            # Integrate expert review
            clinical_results["expert_review"] = self.expert_integrator.integrate_review(
                datasets_results
            )
            
        except Exception as e:
            self.logger.error(f"Error in clinical validation: {e}")
            clinical_results["error"] = str(e)
        
        return clinical_results
    
    def _run_safety_assessment(self, datasets_results: Dict) -> Dict:
        """Run comprehensive safety assessment."""
        safety_results = {
            "compliance_check": {},
            "risk_analysis": {},
            "phi_protection": {}
        }
        
        try:
            all_model_outputs = []
            for dataset_results in datasets_results.values():
                all_model_outputs.extend([s.get("model_output", "") for s in dataset_results.get("sample_results", [])])
            
            # Check safety compliance
            safety_results["compliance_check"] = self.safety_checker.check_compliance(
                all_model_outputs
            )
            
            # Analyze risks
            safety_results["risk_analysis"] = self.safety_checker.analyze_risks(
                all_model_outputs
            )
            
            # Check PHI protection
            safety_results["phi_protection"] = self.data_validator.validate_phi_protection(
                all_model_outputs
            )
            
        except Exception as e:
            self.logger.error(f"Error in safety assessment: {e}")
            safety_results["error"] = str(e)
        
        return safety_results
    
    def _get_model_info(self, model_path: str) -> Dict:
        """Get model information."""
        return {
            "model_path": model_path,
            "model_type": "medical_ai_assistant",
            "evaluation_date": datetime.now().isoformat()
        }
    
    def _generate_performance_summary(self, results: Dict) -> Dict:
        """Generate performance summary."""
        summary = {
            "overall_score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "benchmark_comparison": {}
        }
        
        # Calculate overall score based on key metrics
        metrics = results.get("metrics", {}).get("overall_performance", {})
        
        scores = []
        if "medical_accuracy" in metrics:
            medical_f1 = metrics["medical_accuracy"].get("f1", 0)
            scores.append(medical_f1)
        
        if "clinical_assessment" in metrics:
            clinical_score = metrics["clinical_assessment"].get("avg_quality_score", 0)
            scores.append(clinical_score)
        
        if "safety_assessment" in metrics:
            safety_score = metrics["safety_assessment"].get("avg_safety_score", 0)
            scores.append(safety_score)
        
        if "relevance_scoring" in metrics:
            relevance_score = metrics["relevance_scoring"].get("avg_relevance_score", 0)
            scores.append(relevance_score)
        
        if scores:
            summary["overall_score"] = np.mean(scores)
        
        # Generate recommendations
        if summary["overall_score"] < 0.7:
            summary["recommendations"].append("Model performance needs improvement across all metrics")
        else:
            summary["recommendations"].append("Model shows good overall performance")
        
        if "medical_accuracy" in metrics:
            f1_score = metrics["medical_accuracy"].get("f1", 0)
            if f1_score < 0.8:
                summary["recommendations"].append("Focus on improving medical accuracy")
        
        if "safety_assessment" in metrics:
            safety_score = metrics["safety_assessment"].get("avg_safety_score", 0)
            if safety_score < 0.9:
                summary["recommendations"].append("Prioritize safety improvements")
        
        return summary
    
    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save evaluation results to files."""
        # Save detailed results
        results_file = os.path.join(output_dir, "reports", "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary report
        summary_file = os.path.join(output_dir, "reports", "evaluation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(results["performance_summary"], f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_visualizations(self, results: Dict, output_dir: str) -> None:
        """Generate performance visualizations."""
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # Medical accuracy visualization
            if "medical_accuracy" in results.get("metrics", {}).get("overall_performance", {}):
                self._plot_medical_accuracy(results, viz_dir)
            
            # Safety scores visualization
            self._plot_safety_scores(results, viz_dir)
            
            # Dataset comparison
            self._plot_dataset_comparison(results, viz_dir)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    def _plot_medical_accuracy(self, results: Dict, output_dir: str) -> None:
        """Plot medical accuracy metrics."""
        metrics = results["metrics"]["overall_performance"]["medical_accuracy"]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        categories = ["Precision", "Recall", "F1-Score"]
        values = [metrics.get("precision", 0), metrics.get("recall", 0), metrics.get("f1", 0)]
        
        bars = ax.bar(categories, values, color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_ylim(0, 1)
        ax.set_title('Medical Accuracy Metrics')
        ax.set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'medical_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_safety_scores(self, results: Dict, output_dir: str) -> None:
        """Plot safety assessment scores."""
        safety_data = results["datasets"]
        datasets = list(safety_data.keys())
        safety_scores = []
        
        for dataset_name in datasets:
            dataset_result = safety_data[dataset_name]
            if "safety_assessment" in dataset_result:
                score = dataset_result["safety_assessment"].get("avg_safety_score", 0)
                safety_scores.append(score)
            else:
                safety_scores.append(0)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        bars = ax.bar(datasets, safety_scores, color='lightcoral')
        ax.set_ylim(0, 1)
        ax.set_title('Safety Assessment Scores by Dataset')
        ax.set_ylabel('Safety Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, safety_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'safety_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_comparison(self, results: Dict, output_dir: str) -> None:
        """Plot dataset comparison metrics."""
        datasets = list(results["datasets"].keys())
        
        # Collect metrics for each dataset
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for dataset_name in datasets:
            dataset_result = results["datasets"][dataset_name]
            if "medical_accuracy" in dataset_result:
                precision_scores.append(dataset_result["medical_accuracy"].get("avg_precision", 0))
                recall_scores.append(dataset_result["medical_accuracy"].get("avg_recall", 0))
                f1_scores.append(dataset_result["medical_accuracy"].get("avg_f1", 0))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        # Create comparison plot
        x = np.arange(len(datasets))
        width = 0.25
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall_scores, width, label='Recall', color='lightgreen')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='salmon')
        
        ax.set_xlabel('Datasets')
        ax.set_ylabel('Scores')
        ax.set_title('Medical Accuracy Metrics Comparison Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Comprehensive Medical AI Model Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model to evaluate")
    parser.add_argument("--test_datasets", nargs="+", help="Paths to test datasets")
    parser.add_argument("--config", help="Path to evaluation configuration")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config)
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model_path=args.model_path,
        test_datasets=args.test_datasets,
        output_dir=args.output_dir
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    summary = results["performance_summary"]
    print(f"Overall Score: {summary['overall_score']:.3f}")
    print(f"\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"- {rec}")
    
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()