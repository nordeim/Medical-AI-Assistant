#!/usr/bin/env python3
"""
Data Preprocessing Script for Medical AI Training

This script provides a CLI interface for preprocessing medical conversation data
with comprehensive configuration management, progress tracking, and error handling.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime

# Add training utils to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.preprocessing_pipeline import (
        PreprocessingPipeline, 
        PreprocessingConfig, 
        preprocess_medical_conversations
    )
    from utils.data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline
except ImportError:
    # Fallback for direct execution
    try:
        from preprocessing_pipeline import (
            PreprocessingPipeline, 
            PreprocessingConfig, 
            preprocess_medical_conversations
        )
        from data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline
    except ImportError:
        # Last resort - try relative imports
        from ..utils.preprocessing_pipeline import (
            PreprocessingPipeline, 
            PreprocessingConfig, 
            preprocess_medical_conversations
        )
        from ..utils.data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline


class ProgressTracker:
    """Tracks preprocessing progress and provides reporting"""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.current_stage = ""
        self.stage_start_time = time.time()
        self.errors = []
        self.warnings = []
    
    def update_stage(self, stage_name: str):
        """Update current processing stage"""
        if self.current_stage and self.current_stage != stage_name:
            stage_time = time.time() - self.stage_start_time
            logging.info(f"Completed stage '{self.current_stage}' in {stage_time:.2f}s")
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        logging.info(f"Starting stage: {stage_name}")
    
    def update_progress(self, items_processed: int = 1):
        """Update processing progress"""
        self.processed_items += items_processed
        progress_pct = (self.processed_items / self.total_items) * 100
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        if self.processed_items > 0:
            estimated_total_time = elapsed_time * self.total_items / self.processed_items
            eta = estimated_total_time - elapsed_time
            
            logging.info(f"Progress: {progress_pct:.1f}% ({self.processed_items}/{self.total_items}) - "
                        f"ETA: {eta:.0f}s - Stage: {self.current_stage}")
    
    def add_error(self, error_msg: str):
        """Add error to tracking"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "stage": self.current_stage
        })
    
    def add_warning(self, warning_msg: str):
        """Add warning to tracking"""
        self.warnings.append({
            "timestamp": datetime.now().isoformat(),
            "warning": warning_msg,
            "stage": self.current_stage
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary"""
        total_time = time.time() - self.start_time
        
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "total_time_seconds": total_time,
            "average_time_per_item": total_time / self.processed_items if self.processed_items > 0 else 0,
            "stages_completed": self.current_stage,
            "errors": self.errors,
            "warnings": self.warnings
        }


class ConfigManager:
    """Manages configuration loading and validation"""
    
    @staticmethod
    def load_config(config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        
        if config_file and Path(config_file).exists():
            config_path = Path(config_file)
            logging.info(f"Loading configuration from {config_path}")
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        else:
            logging.info("Using default configuration")
            return {}
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration"""
        
        validated_config = {}
        
        # Preprocessing config
        preprocessing_defaults = {
            "batch_size": 1000,
            "max_workers": 4,
            "enable_streaming": True,
            "cache_intermediate_results": True,
            "output_format": "json",
            "min_conversation_length": 2,
            "max_conversation_length": 50,
            "min_text_length": 5,
            "max_text_length": 1000
        }
        
        preprocessing_config = config.get("preprocessing", {})
        for key, default_value in preprocessing_defaults.items():
            validated_config[key] = preprocessing_config.get(key, default_value)
        
        # Augmentation config
        augmentation_defaults = {
            "enable_augmentation": False,
            "synonym_probability": 0.3,
            "paraphrase_probability": 0.4,
            "max_augmentations": 3,
            "preserve_medical_terms": True,
            "diversity_threshold": 0.8
        }
        
        augmentation_config = config.get("augmentation", {})
        for key, default_value in augmentation_defaults.items():
            validated_config[f"aug_{key}"] = augmentation_config.get(key, default_value)
        
        # Output config
        output_defaults = {
            "output_dir": "./preprocessed_data",
            "include_quality_report": True,
            "include_augmentation_report": True,
            "compress_output": False
        }
        
        output_config = config.get("output", {})
        for key, default_value in output_defaults.items():
            validated_config[key] = output_config.get(key, default_value)
        
        return validated_config
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_file: str):
        """Save configuration to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
        
        logging.info(f"Configuration saved to {output_file}")


class DataProcessor:
    """Main data processing orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocess_config = PreprocessingConfig()
        self.augment_config = AugmentationConfig()
        
        # Apply configuration
        self._apply_config(config)
    
    def _apply_config(self, config: Dict[str, Any]):
        """Apply configuration to components"""
        
        # Apply preprocessing config
        for key, value in config.items():
            if hasattr(self.preprocess_config, key):
                setattr(self.preprocess_config, key, value)
        
        # Apply augmentation config
        for key, value in config.items():
            if key.startswith("aug_"):
                attr_name = key[4:]  # Remove "aug_" prefix
                if hasattr(self.augment_config, attr_name):
                    setattr(self.augment_config, attr_name, value)
    
    def load_data(self, input_file: str) -> Dict[str, Any]:
        """Load input data"""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        logging.info(f"Loading data from {input_file}")
        
        with open(input_path, 'r') as f:
            if input_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported input file format: {input_path.suffix}")
        
        # Extract conversations
        conversations = []
        if isinstance(data, dict):
            if "scenarios" in data:
                conversations = data["scenarios"]
            elif "conversations" in data:
                conversations = data["conversations"]
            else:
                conversations = [data]
        elif isinstance(data, list):
            conversations = data
        
        logging.info(f"Loaded {len(conversations)} conversations")
        return {"scenarios": conversations}
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data"""
        
        logging.info("Starting data preprocessing...")
        progress = ProgressTracker(len(data["scenarios"]))
        progress.update_stage("preprocessing")
        
        # Create preprocessing pipeline
        pipeline = PreprocessingPipeline(self.preprocess_config)
        
        # Preprocess dataset
        preprocessed_data = pipeline.preprocess_dataset(data["scenarios"])
        
        progress.processed_items = len(preprocessed_data["conversations"])
        progress.update_progress(0)  # Final update
        
        # Generate processing report
        processing_report = pipeline.generate_processing_report(preprocessed_data)
        
        return {
            "preprocessed_data": preprocessed_data,
            "processing_report": processing_report,
            "progress_summary": progress.get_summary()
        }
    
    def augment_data(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Augment preprocessed data"""
        
        if not self.config.get("aug_enable_augmentation", False):
            logging.info("Data augmentation disabled")
            return {"augmented_data": None, "augmentation_report": None}
        
        logging.info("Starting data augmentation...")
        scenarios = preprocessed_data["preprocessed_data"]["conversations"]
        progress = ProgressTracker(len(scenarios))
        progress.update_stage("augmentation")
        
        # Create augmentor
        augmentor = DataAugmentor(self.augment_config)
        
        # Apply augmentation pipeline
        augmented_results = apply_augmentation_pipeline(scenarios, self.augment_config, augmentor)
        
        progress.processed_items = len(augmented_results["augmented_conversations"])
        progress.update_progress(0)  # Final update
        
        return {
            "augmented_data": augmented_results,
            "augmentation_report": progress.get_summary()
        }
    
    def save_results(self, 
                    preprocessed_result: Dict[str, Any], 
                    augmented_result: Dict[str, Any] = None,
                    output_dir: str = None) -> Dict[str, str]:
        """Save processing results"""
        
        if output_dir is None:
            output_dir = self.config.get("output_dir", "./preprocessed_data")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save preprocessed data
        preprocessed_file = output_path / f"preprocessed_data_{timestamp}.json"
        with open(preprocessed_file, 'w') as f:
            json.dump(preprocessed_result["preprocessed_data"], f, indent=2, default=str)
        saved_files["preprocessed_data"] = str(preprocessed_file)
        
        # Save processing report
        if self.config.get("output_include_quality_report", True):
            report_file = output_path / f"processing_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(preprocessed_result["processing_report"], f, indent=2, default=str)
            saved_files["processing_report"] = str(report_file)
        
        # Save augmented data if available
        if augmented_result and augmented_result.get("augmented_data"):
            augmented_file = output_path / f"augmented_data_{timestamp}.json"
            with open(augmented_file, 'w') as f:
                json.dump(augmented_result["augmented_data"], f, indent=2, default=str)
            saved_files["augmented_data"] = str(augmented_file)
            
            # Save augmentation report
            if self.config.get("output_include_augmentation_report", True):
                aug_report_file = output_path / f"augmentation_report_{timestamp}.json"
                with open(aug_report_file, 'w') as f:
                    json.dump(augmented_result["augmentation_report"], f, indent=2, default=str)
                saved_files["augmentation_report"] = str(aug_report_file)
        
        # Save configuration used
        config_file = output_path / f"config_{timestamp}.json"
        ConfigManager.save_config(self.config, str(config_file))
        saved_files["configuration"] = str(config_file)
        
        logging.info(f"Results saved to {output_dir}")
        return saved_files


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Preprocess medical conversation data for AI training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.json --output ./preprocessed
  %(prog)s --input data.json --config config.yaml --output ./preprocessed
  %(prog)s --input data.json --enable-augmentation --output ./preprocessed
        """
    )
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", 
                       type=str, 
                       required=True,
                       help="Input JSON file containing medical conversations")
    
    parser.add_argument("--output", "-o",
                       type=str,
                       default="./preprocessed_data",
                       help="Output directory for processed data")
    
    # Configuration
    parser.add_argument("--config", "-c",
                       type=str,
                       help="Configuration file (YAML or JSON)")
    
    parser.add_argument("--save-config",
                       type=str,
                       help="Save default configuration to file")
    
    # Processing options
    parser.add_argument("--enable-augmentation",
                       action="store_true",
                       help="Enable data augmentation")
    
    parser.add_argument("--augmentation-strategy",
                       type=str,
                       default="balanced",
                       choices=["balanced", "conservative", "aggressive"],
                       help="Augmentation strategy")
    
    parser.add_argument("--max-workers",
                       type=int,
                       default=4,
                       help="Number of worker processes")
    
    parser.add_argument("--batch-size",
                       type=int,
                       default=1000,
                       help="Batch size for processing")
    
    # Quality control
    parser.add_argument("--enable-quality-filters",
                       action="store_true",
                       help="Enable quality filtering")
    
    parser.add_argument("--validate-medical-accuracy",
                       action="store_true",
                       help="Validate medical accuracy")
    
    # Logging and reporting
    parser.add_argument("--log-level",
                       type=str,
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    parser.add_argument("--log-file",
                       type=str,
                       help="Log file path")
    
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Enable verbose output")
    
    parser.add_argument("--quiet", "-q",
                       action="store_true",
                       help="Suppress non-essential output")
    
    # Help and version
    parser.add_argument("--version",
                       action="version",
                       version="%(prog)s 1.0.0")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    if args.quiet:
        log_level = "ERROR"
    
    setup_logging(log_level, args.log_file)
    
    try:
        # Load configuration
        config = ConfigManager.load_config(args.config)
        config = ConfigManager.validate_config(config)
        
        # Override with command line arguments
        config.update({
            "output_dir": args.output,
            "aug_enable_augmentation": args.enable_augmentation,
            "max_workers": args.max_workers,
            "batch_size": args.batch_size,
            "enable_quality_filters": args.enable_quality_filters
        })
        
        # Apply augmentation strategy
        if args.enable_augmentation:
            if args.augmentation_strategy == "conservative":
                config.update({
                    "aug_synonym_probability": 0.2,
                    "aug_paraphrase_probability": 0.2,
                    "aug_max_augmentations": 2
                })
            elif args.augmentation_strategy == "aggressive":
                config.update({
                    "aug_synonym_probability": 0.5,
                    "aug_paraphrase_probability": 0.6,
                    "aug_max_augmentations": 5
                })
        
        # Save configuration if requested
        if args.save_config:
            ConfigManager.save_config(config, args.save_config)
            logging.info(f"Default configuration saved to {args.save_config}")
            return
        
        logging.info("Starting data preprocessing pipeline...")
        start_time = time.time()
        
        # Create processor
        processor = DataProcessor(config)
        
        # Load data
        data = processor.load_data(args.input)
        
        # Preprocess data
        preprocessed_result = processor.preprocess_data(data)
        
        # Augment data if enabled
        augmented_result = None
        if args.enable_augmentation:
            augmented_result = processor.augment_data(preprocessed_result["preprocessed_data"])
        
        # Save results
        saved_files = processor.save_results(
            preprocessed_result, 
            augmented_result, 
            args.output
        )
        
        # Print summary
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Conversations processed: {preprocessed_result['progress_summary']['processed_items']}")
        print(f"Conversations filtered: {preprocessed_result['progress_summary']['errors_count']}")
        
        if augmented_result:
            print(f"Augmented conversations: {len(augmented_result['augmented_data']['augmented_conversations'])}")
        
        print("\nOutput files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        if not args.quiet:
            print(f"\nDetailed logs saved to: {args.log_file or 'console'}")
        
        print("="*60)
        
        # Exit with success code
        sys.exit(0)
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()