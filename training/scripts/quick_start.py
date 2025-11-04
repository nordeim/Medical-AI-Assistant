#!/usr/bin/env python3
"""
Medical AI Training Pipeline - Quick Start Script
===============================================

This script provides a one-command setup for the complete training pipeline,
including environment configuration, sample data generation, training execution,
and results evaluation.

Usage:
    python quick_start.py [options]

Author: Medical AI Training Team
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import deepspeed
    import transformers
    from transformers import AutoTokenizer, AutoModel
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    print("✅ All required packages are available")
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

class QuickStartTrainer:
    """
    Comprehensive training pipeline automation script.
    
    Features:
    - Automated environment setup
    - Sample data generation
    - Configuration management
    - Training execution with monitoring
    - Results evaluation and reporting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the QuickStartTrainer."""
        self.config = config or self._get_default_config()
        self.setup_logging()
        self.results_dir = Path("quick_start_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quick start."""
        return {
            "model_name": "microsoft/DialoGPT-small",  # Small model for quick testing
            "output_dir": "./quick_start_output",
            "max_length": 512,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "warmup_steps": 100,
            "logging_steps": 10,
            "eval_steps": 500,
            "save_steps": 500,
            "max_grad_norm": 1.0,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "dataloader_num_workers": 2,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "fp16",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "remove_unused_columns": False,
            "report_to": "none",
            "run_name": f"quick_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('quick_start.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_environment(self) -> bool:
        """
        Check if the environment is properly configured.
        
        Returns:
            bool: True if environment is OK, False otherwise
        """
        self.logger.info("🔍 Checking environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("❌ Python 3.8+ is required")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            self.logger.info(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                self.logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.logger.warning("⚠️  CUDA not available, training will use CPU")
        
        # Check required packages
        required_packages = [
            'torch', 'transformers', 'deepspeed', 'numpy', 'pandas', 'sklearn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                self.logger.error(f"❌ {package} is missing")
        
        if missing_packages:
            self.logger.error(f"❌ Missing packages: {missing_packages}")
            self.logger.info("Please install with: pip install -r requirements.txt")
            return False
        
        # Check available disk space (minimum 10GB)
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        
        if free_gb < 10:
            self.logger.warning(f"⚠️  Only {free_gb}GB free disk space (recommend 10GB+)")
        else:
            self.logger.info(f"✅ Disk space: {free_gb}GB available")
        
        self.logger.info("✅ Environment check completed")
        return True
    
    def generate_sample_data(self) -> str:
        """
        Generate sample medical training data.
        
        Returns:
            str: Path to the generated data file
        """
        self.logger.info("🏥 Generating sample medical training data...")
        
        # Generate sample medical conversations
        conversations = []
        
        # Sample medical scenarios
        medical_scenarios = [
            {
                "patient_id": "P001",
                "age": 45,
                "symptoms": ["fever", "cough", "fatigue"],
                "diagnosis": "common cold",
                "treatment": "rest, fluids, over-the-counter medications",
                "follow_up": "if symptoms worsen after 3 days"
            },
            {
                "patient_id": "P002", 
                "age": 67,
                "symptoms": ["chest pain", "shortness of breath"],
                "diagnosis": "possible cardiac issue",
                "treatment": "immediate cardiac evaluation",
                "follow_up": "urgent cardiology consultation"
            },
            {
                "patient_id": "P003",
                "age": 32,
                "symptoms": ["headache", "nausea", "sensitivity to light"],
                "diagnosis": "migraine",
                "treatment": "pain management, rest in dark room",
                "follow_up": "neurology referral if recurrent"
            },
            {
                "patient_id": "P004",
                "age": 28,
                "symptoms": ["abdominal pain", "nausea", "loss of appetite"],
                "diagnosis": "gastroenteritis",
                "treatment": "dietary modification, hydration",
                "follow_up": "if symptoms persist > 48 hours"
            },
            {
                "patient_id": "P005",
                "age": 55,
                "symptoms": ["joint pain", "stiffness", "fatigue"],
                "diagnosis": "arthritis",
                "treatment": "anti-inflammatory medication, physical therapy",
                "follow_up": "rheumatology consultation"
            }
        ]
        
        # Generate conversations for each scenario
        for i, scenario in enumerate(medical_scenarios * 4):  # Multiply for more data
            conversation = {
                "conversation_id": f"CONV_{i+1:04d}",
                "patient_id": scenario["patient_id"],
                "turns": [
                    {
                        "speaker": "patient",
                        "text": f"I am a {scenario['age']}-year-old patient with {', '.join(scenario['symptoms'])}"
                    },
                    {
                        "speaker": "doctor", 
                        "text": f"I understand you're experiencing {', '.join(scenario['symptoms'])}. Let me ask some questions to better understand your condition."
                    },
                    {
                        "speaker": "patient",
                        "text": "I've been feeling this way for about 2 days now. The symptoms are affecting my daily activities."
                    },
                    {
                        "speaker": "doctor",
                        "text": f"Based on your symptoms, this appears to be {scenario['diagnosis']}. I recommend {scenario['treatment']}."
                    },
                    {
                        "speaker": "doctor",
                        "text": f"Please follow up {scenario['follow_up']}. If you have any concerns, don't hesitate to contact our office."
                    }
                ],
                "diagnosis": scenario["diagnosis"],
                "treatment": scenario["treatment"],
                "metadata": {
                    "age": scenario["age"],
                    "symptoms": scenario["symptoms"],
                    "generated_at": datetime.now().isoformat()
                }
            }
            conversations.append(conversation)
        
        # Save to JSON file
        data_file = self.results_dir / "sample_medical_data.json"
        with open(data_file, 'w') as f:
            json.dump(conversations, f, indent=2)
        
        self.logger.info(f"✅ Generated {len(conversarios)} sample conversations")
        self.logger.info(f"✅ Saved to: {data_file}")
        
        return str(data_file)
    
    def create_training_config(self, data_file: str) -> str:
        """
        Create DeepSpeed configuration for training.
        
        Args:
            data_file: Path to training data
            
        Returns:
            str: Path to configuration file
        """
        self.logger.info("⚙️  Creating training configuration...")
        
        # Determine DeepSpeed configuration based on available resources
        config = {
            "bfloat16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_accumulation_steps": self.config["gradient_accumulation_steps"],
            "gradient_clipping": self.config["max_grad_norm"],
            "steps_per_print": 10,
            "train_batch_size": self.config["batch_size"],
            "train_micro_batch_size_per_gpu": self.config["batch_size"],
            "wall_clock_breakdown": False
        }
        
        # Add ZeRO Stage 3 for larger models
        if torch.cuda.device_count() > 1:
            config["zero_optimization"]["stage"] = 3
            config["zero_optimization"]["gather_16bit_weights_on_model_save"] = True
        
        # Save configuration
        config_file = self.results_dir / "deepspeed_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create training arguments
        training_args = {
            "output_dir": self.config["output_dir"],
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": True,
            "do_predict": False,
            "evaluation_strategy": "steps",
            "eval_steps": self.config["eval_steps"],
            "save_steps": self.config["save_steps"],
            "logging_steps": self.config["logging_steps"],
            "save_total_limit": self.config["save_total_limit"],
            "learning_rate": self.config["learning_rate"],
            "weight_decay": self.config["weight_decay"],
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": self.config["max_grad_norm"],
            "num_train_epochs": self.config["num_epochs"],
            "warmup_steps": self.config["warmup_steps"],
            "warmup_ratio": self.config["warmup_ratio"],
            "logging_dir": str(self.results_dir / "logs"),
            "logging_first_step": False,
            "load_best_model_at_end": self.config["load_best_model_at_end"],
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "run_name": self.config["run_name"],
            "disable_tqdm": False,
            "remove_unused_columns": self.config["remove_unused_columns"],
            "label_names": ["labels"],
            "pad_to_max_length": True,
            "length_column_name": "length",
            "report_to": self.config["report_to"],
            "dataloader_num_workers": self.config["dataloader_num_workers"],
            "dataloader_pin_memory": True,
            "skip_memory_metrics": False,
            "use_ipex": False,
            "torch_compile": False,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default",
            "full_determinism": False,
            "torchdynamo": None,
            "ray_scope": "last",
            "ddp_broadcast_buffers": None,
            "ddp_find_unused_parameters": None,
            "ddp_bucket_cap_mb": None,
            "debug": "",
            "deepspeed": str(config_file),
            "include_inputs_for_metrics": True,
            "include_for_metrics": ["accuracy", "f1"],
            "group_by_length": False,
            "length_column_name": "length",
            "ddp_backend": "nccl",
            "dataloader_persistent_workers": False,
            "dataloader_prefetch_factor": 2,
            "optim": "adamw_torch",
            "optim_args": None,
            "lr_scheduler_type": "linear",
            "lr_scheduler_kwargs": {},
            "ddp_timeout": 1800,
            "torch_compile_options": None,
            "filter_by_length": False,
            "length_column_name": "length",
            "group_by_length": False,
            "dataset_kwargs": None,
            "dataset_map_kwargs": None,
            "dataset_filter_kwargs": None,
            "train_dataset_text_field_name": "text",
            "train_dataset_group_field_name": None,
            "train_dataset_group_key": None,
            "train_dataset_sort_key": None,
            "eval_dataset_text_field_name": "text",
            "eval_dataset_group_field_name": None,
            "eval_dataset_group_key": None,
            "eval_dataset_sort_key": None,
            "predict_dataset_text_field_name": "text",
            "predict_dataset_group_field_name": None,
            "predict_dataset_group_key": None,
            "predict_dataset_sort_key": None,
            "train_new_from_scratch": False,
            "continue_from_checkpoint": None,
            "resume_from_checkpoint": None,
            "ignore_skip_first_metric_updates": None,
            "include_inputs_for_metrics": True
        }
        
        # Save training arguments
        args_file = self.results_dir / "training_args.yaml"
        with open(args_file, 'w') as f:
            yaml.dump(training_args, f, default_flow_style=False)
        
        self.logger.info(f"✅ Created training configuration: {config_file}")
        self.logger.info(f"✅ Created training arguments: {args_file}")
        
        return str(config_file)
    
    def prepare_dataset(self, data_file: str) -> Any:
        """
        Prepare and tokenize the dataset.
        
        Args:
            data_file: Path to training data
            
        Returns:
            Dataset object ready for training
        """
        self.logger.info("📊 Preparing dataset...")
        
        try:
            # Load data
            with open(data_file, 'r') as f:
                conversations = json.load(f)
            
            self.logger.info(f"Loaded {len(conversations)} conversations")
            
            # Prepare text data for training
            texts = []
            for conv in conversations:
                # Combine all turns into a single text
                full_text = ""
                for turn in conv["turns"]:
                    speaker = "Patient" if turn["speaker"] == "patient" else "Doctor"
                    full_text += f"{speaker}: {turn['text']}\n"
                
                # Add metadata
                text_with_metadata = f"{full_text}\nDiagnosis: {conv['diagnosis']}\nTreatment: {conv['treatment']}"
                texts.append(text_with_metadata)
            
            self.logger.info(f"Prepared {len(texts)} text samples")
            
            # Create simple dataset class
            class MedicalDataset(Dataset):
                def __init__(self, texts, tokenizer, max_length=512):
                    self.texts = texts
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        padding='max_length',
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].flatten(),
                        'attention_mask': encoding['attention_mask'].flatten(),
                        'labels': encoding['input_ids'].flatten()
                    }
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_name"],
                use_fast=True
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create datasets
            train_size = int(0.8 * len(texts))
            train_texts = texts[:train_size]
            eval_texts = texts[train_size:]
            
            train_dataset = MedicalDataset(train_texts, tokenizer, self.config["max_length"])
            eval_dataset = MedicalDataset(eval_texts, tokenizer, self.config["max_length"])
            
            self.logger.info(f"✅ Created training dataset: {len(train_dataset)} samples")
            self.logger.info(f"✅ Created evaluation dataset: {len(eval_dataset)} samples")
            
            return train_dataset, eval_dataset, tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing dataset: {e}")
            raise
    
    def train_model(self, train_dataset, eval_dataset, tokenizer) -> bool:
        """
        Execute the training process.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer instance
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        self.logger.info("🚀 Starting model training...")
        
        try:
            # Import after confirming environment
            from transformers import (
                AutoModelForCausalLM, 
                Trainer, 
                TrainingArguments,
                DataCollatorForLanguageModeling
            )
            from datasets import Dataset as HFDataset
            
            # Convert to HuggingFace datasets
            def prepare_hf_dataset(pytorch_dataset):
                texts = []
                for i in range(len(pytorch_dataset)):
                    item = pytorch_dataset[i]
                    texts.append({
                        'input_ids': item['input_ids'].tolist(),
                        'attention_mask': item['attention_mask'].tolist(),
                        'labels': item['labels'].tolist()
                    })
                return HFDataset.from_list(texts)
            
            train_hf_dataset = prepare_hf_dataset(train_dataset)
            eval_hf_dataset = prepare_hf_dataset(eval_dataset)
            
            # Load model
            self.logger.info(f"Loading model: {self.config['model_name']}")
            model = AutoModelForCausalLM.from_pretrained(
                self.config["model_name"],
                torch_dtype=torch.float16 if self.config["mixed_precision"] == "fp16" else torch.float32,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
            
            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=None
            )
            
            # Create training arguments
            training_args = TrainingArguments(
                output_dir=self.config["output_dir"],
                overwrite_output_dir=True,
                do_train=True,
                do_eval=True,
                evaluation_strategy="steps",
                eval_steps=self.config["eval_steps"],
                save_steps=self.config["save_steps"],
                logging_steps=self.config["logging_steps"],
                save_total_limit=self.config["save_total_limit"],
                learning_rate=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
                num_train_epochs=self.config["num_epochs"],
                warmup_steps=self.config["warmup_steps"],
                warmup_ratio=self.config["warmup_ratio"],
                logging_dir=str(self.results_dir / "logs"),
                logging_first_step=False,
                load_best_model_at_end=self.config["load_best_model_at_end"],
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                run_name=self.config["run_name"],
                remove_unused_columns=self.config["remove_unused_columns"],
                label_names=["labels"],
                dataloader_num_workers=self.config["dataloader_num_workers"],
                report_to=self.config["report_to"],
                save_strategy="steps",
                logging_strategy="steps",
                gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                fp16=self.config["mixed_precision"] == "fp16",
                bf16=self.config["mixed_precision"] == "bf16",
                max_grad_norm=self.config["max_grad_norm"],
                ddp_find_unused_parameters=None,
                group_by_length=False,
                length_column_name="length",
                disable_tqdm=False,
                skip_memory_metrics=True,
                include_inputs_for_metrics=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_hf_dataset,
                eval_dataset=eval_hf_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics
            )
            
            # Start training
            self.logger.info("Starting training process...")
            start_time = time.time()
            
            # Train the model
            trainer.train()
            
            # Calculate training time
            training_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {training_time/60:.2f} minutes")
            
            # Save the model
            trainer.save_model()
            tokenizer.save_pretrained(self.config["output_dir"])
            
            self.logger.info(f"✅ Model saved to: {self.config['output_dir']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Convert predictions to token IDs
        predictions = predictions.argmax(axis=-1)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.flatten(), 
            predictions.flatten(), 
            average='weighted',
            zero_division=0
        )
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate training results and generate report.
        
        Returns:
            Dict containing evaluation results
        """
        self.logger.info("📊 Evaluating results...")
        
        results = {
            "training_completed": False,
            "model_path": self.config["output_dir"],
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check if model was saved
            if os.path.exists(self.config["output_dir"]):
                model_files = list(Path(self.config["output_dir"]).glob("*"))
                results["model_files"] = len(model_files)
                results["model_path_exists"] = True
            else:
                results["model_path_exists"] = False
                return results
            
            # Check for training logs
            log_file = self.results_dir / "logs" / "training.log"
            if log_file.exists():
                results["training_log_exists"] = True
            else:
                results["training_log_exists"] = False
            
            # Generate summary report
            report = self._generate_training_report(results)
            results["report"] = report
            
            self.logger.info("✅ Results evaluation completed")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating results: {e}")
            results["error"] = str(e)
            return results
    
    def _generate_training_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive training report."""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("MEDICAL AI TRAINING PIPELINE - QUICK START REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {results['timestamp']}")
        report_lines.append("")
        
        # Environment Information
        report_lines.append("ENVIRONMENT INFORMATION:")
        report_lines.append("-" * 30)
        report_lines.append(f"Python Version: {sys.version}")
        report_lines.append(f"PyTorch Version: {torch.__version__}")
        report_lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            report_lines.append(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                report_lines.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        report_lines.append("")
        
        # Configuration Summary
        report_lines.append("TRAINING CONFIGURATION:")
        report_lines.append("-" * 30)
        config_summary = {
            "Model": self.config["model_name"],
            "Batch Size": self.config["batch_size"],
            "Learning Rate": self.config["learning_rate"],
            "Epochs": self.config["num_epochs"],
            "Max Length": self.config["max_length"],
            "LoRA Rank": self.config["lora_rank"],
            "Mixed Precision": self.config["mixed_precision"]
        }
        
        for key, value in config_summary.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Results Summary
        report_lines.append("TRAINING RESULTS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Training Completed: {results['training_completed']}")
        report_lines.append(f"Model Path Exists: {results.get('model_path_exists', False)}")
        report_lines.append(f"Model Files Generated: {results.get('model_files', 0)}")
        report_lines.append(f"Training Log Exists: {results.get('training_log_exists', False)}")
        report_lines.append("")
        
        # Next Steps
        report_lines.append("NEXT STEPS:")
        report_lines.append("-" * 30)
        report_lines.append("1. Review the training logs for detailed metrics")
        report_lines.append("2. Test the trained model using the serving script")
        report_lines.append("3. Fine-tune hyperparameters if needed")
        report_lines.append("4. Consider using larger models for production")
        report_lines.append("5. Implement clinical evaluation procedures")
        report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_summary(self, results: Dict[str, Any]):
        """Save a summary file with key results."""
        summary_file = self.results_dir / "training_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write(results.get("report", "Training completed"))
        
        self.logger.info(f"✅ Summary saved to: {summary_file}")
    
    def cleanup(self):
        """Clean up temporary files."""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        # Clean up any temporary files if needed
        # (Keeping logs and results for reference)
        
        self.logger.info("✅ Cleanup completed")
    
    def run_quick_start(self) -> bool:
        """
        Execute the complete quick start pipeline.
        
        Returns:
            bool: True if all steps completed successfully
        """
        self.logger.info("🎯 Starting Medical AI Training Pipeline - Quick Start")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        success = True
        
        try:
            # Step 1: Environment check
            if not self.check_environment():
                return False
            
            # Step 2: Generate sample data
            data_file = self.generate_sample_data()
            
            # Step 3: Create training configuration
            config_file = self.create_training_config(data_file)
            
            # Step 4: Prepare dataset
            train_dataset, eval_dataset, tokenizer = self.prepare_dataset(data_file)
            
            # Step 5: Train model
            training_success = self.train_model(train_dataset, eval_dataset, tokenizer)
            if not training_success:
                success = False
            
            # Step 6: Evaluate results
            results = self.evaluate_results()
            results["training_completed"] = training_success
            
            # Step 7: Save summary
            self.save_summary(results)
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️  Training interrupted by user")
            success = False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {e}")
            success = False
        finally:
            # Calculate total time
            total_time = time.time() - start_time
            self.logger.info(f"⏱️  Total execution time: {total_time/60:.2f} minutes")
            
            # Cleanup
            self.cleanup()
        
        if success:
            self.logger.info("🎉 Quick start completed successfully!")
            self.logger.info(f"📊 Results saved in: {self.results_dir}")
        else:
            self.logger.error("❌ Quick start failed. Check logs for details.")
        
        return success


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Medical AI Training Pipeline - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py                           # Run with default settings
  python quick_start.py --model-name bert-base-uncased  # Use specific model
  python quick_start.py --epochs 5               # Train for more epochs
  python quick_start.py --batch-size 8           # Increase batch size
        """
    )
    
    # Training configuration arguments
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="microsoft/DialoGPT-small",
        help="Hugging Face model name to use (default: microsoft/DialoGPT-small)"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./quick_start_output",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=4,
        help="Training batch size (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--lora-rank", 
        type=int, 
        default=16,
        help="LoRA rank for parameter-efficient fine-tuning (default: 16)"
    )
    
    parser.add_argument(
        "--mixed-precision", 
        type=str, 
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="Mixed precision training mode (default: fp16)"
    )
    
    parser.add_argument(
        "--no-cuda", 
        action="store_true",
        help="Disable CUDA even if available"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce logging verbosity"
    )
    
    parser.add_argument(
        "--skip-env-check", 
        action="store_true",
        help="Skip environment validation"
    )
    
    parser.add_argument(
        "--config-file", 
        type=str,
        help="Path to JSON configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    config.update({
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_rank": args.lora_rank,
        "mixed_precision": args.mixed_precision
    })
    
    # Override CUDA setting
    if args.no_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Adjust logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create and run trainer
    trainer = QuickStartTrainer(config)
    
    # Skip environment check if requested
    if args.skip_env_check:
        trainer.logger.info("⏭️  Skipping environment check")
    else:
        if not trainer.check_environment():
            sys.exit(1)
    
    # Run the quick start
    success = trainer.run_quick_start()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()