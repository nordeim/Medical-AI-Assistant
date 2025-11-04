#!/usr/bin/env python3
"""
Synthetic Data Generation Script for Medical AI Training

This script provides a comprehensive CLI interface for generating synthetic medical
conversations and datasets for training augmentation.

Usage:
    python generate_synthetic_data.py generate --num-scenarios 1000
    python generate_synthetic_data.py augment --input-file data.json --output-file augmented.json
    python generate_synthetic_data.py validate --input-file synthetic_data.json
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import asdict

# Add training utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from synthetic_data_generator import (
    SyntheticDataGenerator, 
    save_scenarios_to_json, 
    save_scenarios_to_csv,
    MedicalSpecialty,
    TriageLevel
)
from data_augmentation import (
    DataAugmentor, 
    AugmentationConfig,
    apply_augmentation_pipeline,
    save_augmented_conversations
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SyntheticDataGeneratorCLI:
    """CLI interface for synthetic data generation"""
    
    def __init__(self):
        self.output_dir = Path("synthetic_data_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_config = {
            "num_scenarios": 100,
            "age_range": [18, 80],
            "seed": 42,
            "specialty_distribution": {
                "general": 0.3,
                "cardiology": 0.15,
                "respiratory": 0.15,
                "gastrointestinal": 0.1,
                "neurology": 0.1,
                "pediatrics": 0.1,
                "geriatrics": 0.05,
                "orthopedics": 0.025,
                "dermatology": 0.025,
                "mental_health": 0.025
            },
            "triage_distribution": {
                "NON_URGENT": 0.4,
                "LESS_URGENT": 0.3,
                "URGENT": 0.2,
                "EMERGENT": 0.08,
                "IMMEDIATE": 0.02
            }
        }
        
        self.default_augmentation_config = {
            "synonym_probability": 0.3,
            "paraphrase_probability": 0.4,
            "adversarial_probability": 0.2,
            "diversity_threshold": 0.8,
            "max_augmentations": 5,
            "preserve_medical_terms": True,
            "context_aware": True
        }
    
    def generate_data(self, args):
        """Generate synthetic medical data"""
        logger.info(f"Generating {args.num_scenarios} synthetic medical scenarios...")
        
        # Initialize generator
        generator = SyntheticDataGenerator(seed=args.seed)
        
        # Process specialty distribution
        specialty_dist = None
        if args.specialty_distribution:
            specialty_dist = self._parse_distribution(args.specialty_distribution, MedicalSpecialty)
        
        # Process triage distribution
        triage_dist = None
        if args.triage_distribution:
            triage_dist = self._parse_distribution(args.triage_distribution, TriageLevel)
        
        try:
            # Generate scenarios
            scenarios = generator.generate_dataset(
                num_scenarios=args.num_scenarios,
                specialty_distribution=specialty_dist,
                age_range=args.age_range,
                triage_distribution=triage_dist
            )
            
            logger.info(f"Generated {len(scenarios)} scenarios successfully")
            
            # Save outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if args.output_format in ["json", "both"]:
                json_path = self.output_dir / f"synthetic_medical_data_{timestamp}.json"
                save_scenarios_to_json(scenarios, str(json_path))
                logger.info(f"Saved JSON data to {json_path}")
            
            if args.output_format in ["csv", "both"]:
                csv_path = self.output_dir / f"synthetic_medical_data_{timestamp}.csv"
                save_scenarios_to_csv(scenarios, str(csv_path))
                logger.info(f"Saved CSV data to {csv_path}")
            
            # Generate HuggingFace dataset if requested
            if args.huggingface_dataset:
                self._save_as_hf_dataset(scenarios, f"synthetic_medical_data_{timestamp}")
            
            # Generate quality report
            quality_report = self._generate_quality_report(scenarios)
            report_path = self.output_dir / f"quality_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            
            logger.info(f"Generated quality report: {report_path}")
            
            # Print summary
            self._print_generation_summary(scenarios, quality_report)
            
        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            raise
    
    def augment_data(self, args):
        """Augment existing synthetic data"""
        logger.info(f"Augmenting data from {args.input_file}...")
        
        # Load original data
        with open(args.input_file, 'r') as f:
            original_data = json.load(f)
        
        # Parse augmentation config
        augmentation_config = self._create_augmentation_config(args)
        
        try:
            # Apply augmentation pipeline
            results = apply_augmentation_pipeline(
                original_data.get("scenarios", original_data.get("conversations", [])),
                augmentation_config
            )
            
            logger.info(f"Augmentation complete: {results['original_conversations']} -> {results['augmented_conversations']} conversations")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"augmented_data_{timestamp}.json"
            
            augmentation_results = {
                "original_data": original_data,
                "augmented_conversations": results["augmented_conversations"],
                "quality_metrics": results["quality_metrics"],
                "adversarial_count": results["adversarial_count"],
                "generation_timestamp": timestamp
            }
            
            save_augmented_conversations(results["augmented_conversations"], str(output_path))
            logger.info(f"Saved augmented data to {output_path}")
            
            # Generate augmentation report
            self._generate_augmentation_report(results, output_path.with_suffix('.report.json'))
            
            print("\n" + "="*60)
            print("AUGMENTATION SUMMARY")
            print("="*60)
            print(f"Original conversations: {results['original_conversations']}")
            print(f"Augmented conversations: {len(results['augmented_conversations'])}")
            print(f"Quality metrics: {results['quality_metrics']}")
            print(f"Adversarial examples: {results['adversarial_count']}")
            
        except Exception as e:
            logger.error(f"Error augmenting data: {str(e)}")
            raise
    
    def validate_data(self, args):
        """Validate synthetic data quality"""
        logger.info(f"Validating data from {args.input_file}...")
        
        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)
            
            # Run validation checks
            validation_results = self._validate_dataset(data, args)
            
            # Print results
            self._print_validation_results(validation_results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"validation_report_{timestamp}.json"
            
            with open(report_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise
    
    def batch_generate(self, args):
        """Batch generation of multiple datasets"""
        logger.info(f"Starting batch generation with {args.num_batches} batches...")
        
        base_seed = args.seed
        batch_size = args.batch_size
        
        for batch_idx in range(args.num_batches):
            logger.info(f"Generating batch {batch_idx + 1}/{args.num_batches}")
            
            # Update seed for reproducibility
            current_seed = base_seed + batch_idx
            generator = SyntheticDataGenerator(seed=current_seed)
            
            try:
                # Generate batch
                scenarios = generator.generate_dataset(
                    num_scenarios=batch_size,
                    age_range=args.age_range
                )
                
                # Save batch
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_filename = f"batch_{batch_idx+1:03d}_{timestamp}"
                
                if args.output_format in ["json", "both"]:
                    json_path = self.output_dir / f"{batch_filename}.json"
                    save_scenarios_to_json(scenarios, str(json_path))
                
                if args.output_format in ["csv", "both"]:
                    csv_path = self.output_dir / f"{batch_filename}.csv"
                    save_scenarios_to_csv(scenarios, str(csv_path))
                
                logger.info(f"Batch {batch_idx + 1} saved successfully")
                
            except Exception as e:
                logger.error(f"Error generating batch {batch_idx + 1}: {str(e)}")
                continue
        
        # Generate batch summary
        self._generate_batch_summary(args.num_batches)
    
    def _parse_distribution(self, distribution_str: str, enum_class):
        """Parse distribution string into enum dictionary"""
        distribution = {}
        items = distribution_str.split(',')
        
        for item in items:
            if '=' in item:
                key, value = item.split('=')
                try:
                    enum_value = enum_class(key.strip())
                    distribution[enum_value] = float(value.strip())
                except ValueError:
                    logger.warning(f"Invalid {enum_class.__name__} value: {key.strip()}")
        
        return distribution if distribution else None
    
    def _create_augmentation_config(self, args) -> AugmentationConfig:
        """Create augmentation config from args"""
        config = AugmentationConfig()
        
        if args.synonym_probability:
            config.synonym_probability = args.synonym_probability
        if args.paraphrase_probability:
            config.paraphrase_probability = args.paraphrase_probability
        if args.adversarial_probability:
            config.adversarial_probability = args.adversarial_probability
        if args.max_augmentations:
            config.max_augmentations = args.max_augmentations
        
        return config
    
    def _save_as_hf_dataset(self, scenarios: List, dataset_name: str):
        """Save as HuggingFace dataset"""
        try:
            from datasets import Dataset, DatasetDict
            
            # Convert scenarios to format suitable for HF datasets
            hf_data = []
            for scenario in scenarios:
                for turn in scenario.conversation:
                    hf_data.append({
                        "text": turn.text,
                        "speaker": turn.speaker,
                        "timestamp": turn.timestamp,
                        "scenario_id": scenario.scenario_id,
                        "triage_level": scenario.triage_level.value,
                        "specialty": scenario.specialty.value
                    })
            
            # Create dataset
            dataset = Dataset.from_list(hf_data)
            
            # Save locally
            save_path = self.output_dir / dataset_name
            dataset.save_to_disk(str(save_path))
            
            logger.info(f"Saved HuggingFace dataset to {save_path}")
            
        except ImportError:
            logger.warning("HuggingFace datasets not available. Install with: pip install datasets")
        except Exception as e:
            logger.error(f"Error saving HF dataset: {str(e)}")
    
    def _generate_quality_report(self, scenarios: List) -> Dict[str, Any]:
        """Generate quality report for generated scenarios"""
        
        # Analyze distributions
        specialty_dist = {}
        triage_dist = {}
        age_distribution = {}
        
        for scenario in scenarios:
            specialty = scenario.specialty.value
            triage = scenario.triage_level.value
            age_group = scenario.patient.age_group.value
            
            specialty_dist[specialty] = specialty_dist.get(specialty, 0) + 1
            triage_dist[triage] = triage_dist.get(triage, 0) + 1
            age_distribution[age_group] = age_distribution.get(age_group, 0) + 1
        
        # Calculate metrics
        total_scenarios = len(scenarios)
        avg_conversation_length = sum(len(scenario.conversation) for scenario in scenarios) / total_scenarios
        
        # Medical complexity analysis
        complexity_scores = []
        for scenario in scenarios:
            complexity = len(scenario.symptoms) + (scenario.patient.age // 20)
            complexity_scores.append(complexity)
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores)
        
        return {
            "total_scenarios": total_scenarios,
            "specialty_distribution": specialty_dist,
            "triage_distribution": triage_dist,
            "age_distribution": age_distribution,
            "average_conversation_length": avg_conversation_length,
            "average_complexity_score": avg_complexity,
            "generation_timestamp": datetime.now().isoformat(),
            "quality_indicators": {
                "diversity_score": len(specialty_dist) / 10,  # Theoretical max is 10 specialties
                "balanced_triage": len(triage_dist) >= 4,  # Good if we have 4+ triage levels
                "age_distribution": len(age_distribution) >= 5  # Good if we have 5+ age groups
            }
        }
    
    def _generate_augmentation_report(self, results: Dict, report_path: Path):
        """Generate augmentation quality report"""
        
        report = {
            "augmentation_summary": {
                "original_conversations": results["original_conversations"],
                "total_augmented": len(results["augmented_conversations"]),
                "augmentation_ratio": len(results["augmented_conversations"]) / results["original_conversations"],
                "adversarial_examples": results["adversarial_count"]
            },
            "quality_metrics": results["quality_metrics"],
            "recommendations": self._generate_augmentation_recommendations(results["quality_metrics"]),
            "generation_timestamp": datetime.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _generate_augmentation_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality metrics"""
        recommendations = []
        
        if metrics.get("diversity_score", 0) < 0.3:
            recommendations.append("Consider increasing synonym and paraphrase probabilities for better diversity")
        
        if metrics.get("semantic_similarity", 0) > 0.9:
            recommendations.append("High semantic similarity suggests need for more aggressive augmentation")
        
        if metrics.get("medical_accuracy", 0) < 0.8:
            recommendations.append("Medical accuracy is low - review medical terminology handling")
        
        if not recommendations:
            recommendations.append("Augmentation quality is good. No specific recommendations.")
        
        return recommendations
    
    def _validate_dataset(self, data: Dict[str, Any], args) -> Dict[str, Any]:
        """Validate dataset for quality and completeness"""
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": [],
            "overall_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        scenarios = data.get("scenarios", data.get("conversations", []))
        
        # Basic checks
        validation_results["checks"].append(self._check_dataset_size(scenarios))
        validation_results["checks"].append(self._check_conversation_structure(scenarios))
        validation_results["checks"].append(self._check_medical_terminology(scenarios))
        validation_results["checks"].append(self._check_diversity(scenarios))
        
        # Calculate overall score
        passed_checks = sum(1 for check in validation_results["checks"] if check["passed"])
        validation_results["overall_score"] = passed_checks / len(validation_results["checks"])
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_validation_recommendations(validation_results["checks"])
        
        return validation_results
    
    def _check_dataset_size(self, scenarios: List) -> Dict[str, Any]:
        """Check if dataset size is adequate"""
        size = len(scenarios)
        passed = size >= 50  # Minimum 50 scenarios
        
        return {
            "check_name": "dataset_size",
            "description": "Check if dataset has adequate number of scenarios",
            "passed": passed,
            "value": size,
            "threshold": 50,
            "message": f"Dataset has {size} scenarios" + (" (adequate)" if passed else " (too small)")
        }
    
    def _check_conversation_structure(self, scenarios: List) -> Dict[str, Any]:
        """Check conversation structure quality"""
        well_formed = 0
        
        for scenario in scenarios:
            if isinstance(scenario, dict) and "conversation" in scenario:
                conversation = scenario["conversation"]
                if (len(conversation) >= 4 and 
                    any(turn.get("speaker") == "patient" for turn in conversation) and
                    any(turn.get("speaker") == "ai" for turn in conversation)):
                    well_formed += 1
        
        total = len(scenarios)
        ratio = well_formed / total if total > 0 else 0
        passed = ratio >= 0.8
        
        return {
            "check_name": "conversation_structure",
            "description": "Check if conversations have proper structure",
            "passed": passed,
            "value": ratio,
            "threshold": 0.8,
            "message": f"{ratio:.1%} of conversations are well-formed" + (" (good)" if passed else " (needs improvement)")
        }
    
    def _check_medical_terminology(self, scenarios: List) -> Dict[str, Any]:
        """Check if medical terminology is present"""
        medical_terms = {"pain", "symptom", "diagnosis", "treatment", "medication", "doctor", "hospital"}
        scenarios_with_medical_terms = 0
        
        for scenario in scenarios:
            if isinstance(scenario, dict) and "conversation" in scenario:
                text = " ".join(turn.get("text", "") for turn in scenario["conversation"])
                text_lower = text.lower()
                
                if any(term in text_lower for term in medical_terms):
                    scenarios_with_medical_terms += 1
        
        total = len(scenarios)
        ratio = scenarios_with_medical_terms / total if total > 0 else 0
        passed = ratio >= 0.6
        
        return {
            "check_name": "medical_terminology",
            "description": "Check if conversations contain medical terminology",
            "passed": passed,
            "value": ratio,
            "threshold": 0.6,
            "message": f"{ratio:.1%} of conversations contain medical terms" + (" (good)" if passed else " (needs improvement)")
        }
    
    def _check_diversity(self, scenarios: List) -> Dict[str, Any]:
        """Check diversity of scenarios"""
        if not scenarios:
            return {"check_name": "diversity", "passed": False, "value": 0, "message": "No scenarios to analyze"}
        
        # Check specialty diversity
        specialties = set()
        triage_levels = set()
        
        for scenario in scenarios:
            if isinstance(scenario, dict):
                if "specialty" in scenario:
                    specialties.add(scenario["specialty"])
                if "triage_level" in scenario:
                    triage_levels.add(scenario["triage_level"])
        
        specialty_diversity = len(specialties) / 10  # Theoretical max is 10
        triage_diversity = len(triage_levels) / 5  # Theoretical max is 5
        
        overall_diversity = (specialty_diversity + triage_diversity) / 2
        passed = overall_diversity >= 0.3
        
        return {
            "check_name": "diversity",
            "description": "Check diversity of specialties and triage levels",
            "passed": passed,
            "value": overall_diversity,
            "threshold": 0.3,
            "message": f"Diversity score: {overall_diversity:.1%} (specialties: {len(specialties)}, triage: {len(triage_levels)})" +
                      (" (good)" if passed else " (needs improvement)")
        }
    
    def _generate_validation_recommendations(self, checks: List[Dict]) -> List[str]:
        """Generate validation recommendations"""
        recommendations = []
        
        failed_checks = [check for check in checks if not check["passed"]]
        
        for check in failed_checks:
            if check["check_name"] == "dataset_size":
                recommendations.append("Consider generating more scenarios for better training data")
            elif check["check_name"] == "conversation_structure":
                recommendations.append("Review conversation generation templates to improve structure")
            elif check["check_name"] == "medical_terminology":
                recommendations.append("Enhance medical vocabulary in generation templates")
            elif check["check_name"] == "diversity":
                recommendations.append("Adjust specialty and triage distributions for better diversity")
        
        if not recommendations:
            recommendations.append("All validation checks passed. Dataset quality is good!")
        
        return recommendations
    
    def _print_generation_summary(self, scenarios: List, quality_report: Dict):
        """Print generation summary"""
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"Total scenarios generated: {len(scenarios)}")
        print(f"Specialties represented: {len(quality_report['specialty_distribution'])}")
        print(f"Triage levels represented: {len(quality_report['triage_distribution'])}")
        print(f"Average conversation length: {quality_report['average_conversation_length']:.1f} turns")
        print(f"Average complexity score: {quality_report['average_complexity_score']:.1f}")
        
        print("\nSpecialty Distribution:")
        for specialty, count in quality_report['specialty_distribution'].items():
            percentage = (count / len(scenarios)) * 100
            print(f"  {specialty}: {count} ({percentage:.1f}%)")
    
    def _print_validation_results(self, results: Dict):
        """Print validation results"""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Overall Quality Score: {results['overall_score']:.1%}")
        print("\nIndividual Checks:")
        
        for check in results["checks"]:
            status = "✓ PASS" if check["passed"] else "✗ FAIL"
            print(f"  {status}: {check['description']}")
            print(f"    {check['message']}")
        
        if results["recommendations"]:
            print("\nRecommendations:")
            for i, rec in enumerate(results["recommendations"], 1):
                print(f"  {i}. {rec}")
    
    def _generate_batch_summary(self, num_batches: int):
        """Generate batch processing summary"""
        batch_files = list(self.output_dir.glob("batch_*.json"))
        
        print("\n" + "="*60)
        print("BATCH GENERATION SUMMARY")
        print("="*60)
        print(f"Generated batches: {len(batch_files)}")
        print(f"Output directory: {self.output_dir}")
        
        total_scenarios = 0
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    data = json.load(f)
                    scenarios = data.get("scenarios", [])
                    total_scenarios += len(scenarios)
            except Exception as e:
                logger.error(f"Error reading {batch_file}: {str(e)}")
        
        print(f"Total scenarios across all batches: {total_scenarios}")


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator for Medical AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 synthetic scenarios
  python generate_synthetic_data.py generate --num-scenarios 1000
  
  # Generate with specific distribution
  python generate_synthetic_data.py generate --num-scenarios 500 --specialty-distribution "general=0.5,cardiology=0.2,respiratory=0.3"
  
  # Augment existing data
  python generate_synthetic_data.py augment --input-file synthetic_data.json --synonym-probability 0.4
  
  # Validate data quality
  python generate_synthetic_data.py validate --input-file synthetic_data.json
  
  # Batch generation
  python generate_synthetic_data.py batch --num-batches 5 --batch-size 200
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate synthetic medical data')
    generate_parser.add_argument('--num-scenarios', type=int, default=100,
                                help='Number of scenarios to generate')
    generate_parser.add_argument('--age-range', nargs=2, type=int, default=[18, 80],
                                help='Age range for patients (min max)')
    generate_parser.add_argument('--seed', type=int, default=42,
                                help='Random seed for reproducibility')
    generate_parser.add_argument('--specialty-distribution', type=str,
                                help='Comma-separated specialty distribution (e.g., "general=0.3,cardiology=0.2")')
    generate_parser.add_argument('--triage-distribution', type=str,
                                help='Comma-separated triage distribution')
    generate_parser.add_argument('--output-format', choices=['json', 'csv', 'both'], default='json',
                                help='Output format')
    generate_parser.add_argument('--huggingface-dataset', action='store_true',
                                help='Also save as HuggingFace dataset')
    
    # Augment command
    augment_parser = subparsers.add_parser('augment', help='Augment existing data')
    augment_parser.add_argument('--input-file', required=True,
                               help='Input JSON file with synthetic data')
    augment_parser.add_argument('--synonym-probability', type=float,
                               help='Probability of synonym replacement')
    augment_parser.add_argument('--paraphrase-probability', type=float,
                               help='Probability of paraphrase generation')
    augment_parser.add_argument('--adversarial-probability', type=float,
                               help='Probability of adversarial example generation')
    augment_parser.add_argument('--max-augmentations', type=int,
                               help='Maximum number of augmentations per conversation')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data quality')
    validate_parser.add_argument('--input-file', required=True,
                                help='Input file to validate')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch generate multiple datasets')
    batch_parser.add_argument('--num-batches', type=int, required=True,
                             help='Number of batches to generate')
    batch_parser.add_argument('--batch-size', type=int, default=100,
                             help='Number of scenarios per batch')
    batch_parser.add_argument('--age-range', nargs=2, type=int, default=[18, 80],
                             help='Age range for patients (min max)')
    batch_parser.add_argument('--seed', type=int, default=42,
                             help='Base random seed')
    batch_parser.add_argument('--output-format', choices=['json', 'csv', 'both'], default='json',
                             help='Output format')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = SyntheticDataGeneratorCLI()
    
    try:
        if args.command == 'generate':
            cli.generate_data(args)
        elif args.command == 'augment':
            cli.augment_data(args)
        elif args.command == 'validate':
            cli.validate_data(args)
        elif args.command == 'batch':
            cli.batch_generate(args)
        
        logger.info("Operation completed successfully")
        
    except Exception as e:
        logger.error(f"Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()