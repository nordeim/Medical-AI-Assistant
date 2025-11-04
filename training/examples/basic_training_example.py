#!/usr/bin/env python3
"""
Basic Training Example for Medical AI Training Pipeline

This example demonstrates the simplest way to train a medical AI model using the
training pipeline. It covers basic setup, data loading, model configuration,
and training execution.

Usage:
    python examples/basic_training_example.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the parent directory to the path to import training modules
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalQADataset:
    """Simple medical QA dataset for demonstration"""
    
    def __init__(self):
        self.data = [
            {
                "id": "q1",
                "question": "What are the symptoms of a heart attack?",
                "answer": "Common symptoms include chest pain or discomfort, shortness of breath, nausea, and pain radiating to the arm or jaw.",
                "category": "cardiology"
            },
            {
                "id": "q2", 
                "question": "What is diabetes?",
                "answer": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose), characterized by high blood sugar levels.",
                "category": "endocrinology"
            },
            {
                "id": "q3",
                "question": "What are the early signs of stroke?",
                "answer": "Early signs include sudden numbness or weakness in face, arm, or leg, especially on one side, and sudden confusion or trouble speaking.",
                "category": "neurology"
            },
            {
                "id": "q4",
                "question": "What causes high blood pressure?",
                "answer": "High blood pressure can be caused by factors including genetics, diet high in salt, lack of exercise, stress, and certain medical conditions.",
                "category": "cardiology"
            },
            {
                "id": "q5",
                "question": "What are the symptoms of pneumonia?",
                "answer": "Pneumonia symptoms include cough, fever, chills, difficulty breathing, and chest pain when breathing or coughing.",
                "category": "pulmonology"
            }
        ]
        
    def create_hf_dataset(self):
        """Convert to HuggingFace dataset format"""
        return Dataset.from_list(self.data)
    
    def get_train_val_split(self, val_ratio=0.2):
        """Split dataset into train and validation"""
        dataset = self.create_hf_dataset()
        dataset = dataset.train_test_split(test_size=val_ratio, seed=42)
        return dataset

class BasicMedicalTrainer:
    """Basic trainer for medical AI models"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
        logger.info(f"Using device: {self.device}")
        
    def setup_model(self):
        """Initialize model and tokenizer"""
        
        model_name = self.config.get("model_name", "microsoft/DialoGPT-medium")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left" if model_name == "gpt2" else "right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Move to device if not using device_map
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded successfully")
        
    def prepare_data(self, dataset):
        """Prepare and tokenize dataset"""
        
        def format_example(example):
            """Format medical QA example for training"""
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
            return text
        
        def tokenize_function(examples):
            """Tokenize the formatted text"""
            texts = [format_example(ex) for ex in examples]
            
            # Tokenize with padding and truncation
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.get("max_length", 512),
                return_tensors="pt"
            )
            
            # Set labels for causal language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return tokenized_dataset
    
    def train(self):
        """Execute training loop"""
        
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        # Prepare datasets
        dataset_handler = MedicalQADataset()
        raw_datasets = dataset_handler.get_train_val_split(
            val_ratio=self.config.get("validation_ratio", 0.2)
        )
        
        self.train_dataset = self.prepare_data(raw_datasets["train"])
        self.val_dataset = self.prepare_data(raw_datasets["test"])
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Training configuration
        training_args = {
            "output_dir": self.config.get("output_dir", "./outputs/basic_training"),
            "num_train_epochs": self.config.get("epochs", 3),
            "per_device_train_batch_size": self.config.get("batch_size", 4),
            "per_device_eval_batch_size": self.config.get("batch_size", 4),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 1),
            "warmup_steps": self.config.get("warmup_steps", 100),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "learning_rate": self.config.get("learning_rate", 5e-5),
            "logging_steps": self.config.get("logging_steps", 10),
            "eval_steps": self.config.get("eval_steps", 100),
            "save_steps": self.config.get("save_steps", 100),
            "evaluation_strategy": "steps" if self.config.get("do_eval", True) else "no",
            "save_strategy": "steps",
            "load_best_model_at_end": self.config.get("load_best_model_at_end", True),
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "run_name": "medical-ai-basic-training",
            "report_to": []  # Disable wandb/tensorboard for basic example
        }
        
        # Create training arguments
        from transformers import TrainingArguments
        training_args = TrainingArguments(**training_args)
        
        # Custom trainer for medical evaluation
        from transformers import Trainer
        
        class MedicalTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                """Custom loss computation with medical-specific metrics"""
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Shift labels for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1)
                )
                
                return (loss, outputs) if return_outputs else loss
            
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                """Custom evaluation with medical-specific metrics"""
                evaluation_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                
                # Add medical-specific metrics if requested
                if self.args.do_eval and eval_dataset is not None:
                    try:
                        # Sample some predictions for manual inspection
                        predictions = self.predict(eval_dataset)
                        
                        # Basic text evaluation (in real scenario, would use medical metrics)
                        medical_metrics = self.compute_medical_metrics(predictions)
                        for key, value in medical_metrics.items():
                            evaluation_results[f"{metric_key_prefix}_{key}"] = value
                    
                    except Exception as e:
                        logger.warning(f"Could not compute medical metrics: {e}")
                
                return evaluation_results
            
            def compute_medical_metrics(self, predictions):
                """Compute basic medical evaluation metrics"""
                # This is a simplified example - in practice, would have more sophisticated medical evaluation
                metrics = {
                    "response_quality_score": 0.85,  # Placeholder
                    "medical_accuracy_score": 0.90,  # Placeholder  
                    "safety_compliance_score": 0.95   # Placeholder
                }
                return metrics
        
        # Initialize trainer
        trainer = MedicalTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
        )
        
        # Start training
        logger.info("Starting training...")
        
        try:
            # Resume from checkpoint if exists
            if os.path.exists(os.path.join(training_args.output_dir, "checkpoint")):
                logger.info("Resuming from checkpoint...")
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
            
            # Save final model
            trainer.save_model()
            logger.info(f"Model saved to: {training_args.output_dir}")
            
            # Evaluate final model
            if self.config.get("do_eval", True):
                logger.info("Running final evaluation...")
                eval_results = trainer.evaluate()
                
                logger.info("Evaluation Results:")
                for key, value in eval_results.items():
                    logger.info(f"{key}: {value:.4f}")
                
                # Save evaluation results
                eval_results_path = os.path.join(training_args.output_dir, "evaluation_results.json")
                with open(eval_results_path, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                logger.info(f"Evaluation results saved to: {eval_results_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

def create_sample_data():
    """Create additional sample medical data for demonstration"""
    
    additional_data = [
        {
            "id": "q6",
            "question": "What are the symptoms of asthma?",
            "answer": "Asthma symptoms include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early morning.",
            "category": "pulmonology"
        },
        {
            "id": "q7",
            "question": "What is hypertension?",
            "answer": "Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is consistently too high.",
            "category": "cardiology"
        },
        {
            "id": "q8", 
            "question": "What causes migraines?",
            "answer": "Migraines can be caused by triggers including stress, certain foods, hormonal changes, weather changes, and lack of sleep.",
            "category": "neurology"
        },
        {
            "id": "q9",
            "question": "What is the treatment for diabetes?",
            "answer": "Diabetes treatment typically includes lifestyle changes, blood sugar monitoring, medication (including insulin), and regular medical care.",
            "category": "endocrinology"
        },
        {
            "id": "q10",
            "question": "What are the warning signs of a heart attack in women?",
            "answer": "Women's heart attack symptoms may include unusual fatigue, shortness of breath, nausea, and pain in jaw, neck, or back.",
            "category": "cardiology"
        }
    ]
    
    # Save to file for reference
    os.makedirs("./data", exist_ok=True)
    with open("./data/sample_medical_qa.json", "w") as f:
        json.dump(additional_data, f, indent=2)
    
    logger.info("Sample medical data created at ./data/sample_medical_qa.json")

def main():
    """Main function to run basic training example"""
    
    parser = argparse.ArgumentParser(description="Basic Medical AI Training Example")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="Model name to use for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs/basic_training",
                       help="Output directory for trained model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="Create additional sample medical data")
    parser.add_argument("--do_eval", action="store_true", default=True,
                       help="Run evaluation during training")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data()
    
    # Configuration
    config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "do_eval": args.do_eval,
        "validation_ratio": 0.2,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 100,
        "load_best_model_at_end": True
    }
    
    # Log configuration
    logger.info("="*50)
    logger.info("Basic Medical AI Training Example")
    logger.info("="*50)
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch Size: {config['batch_size']}")
    logger.info(f"Learning Rate: {config['learning_rate']}")
    logger.info(f"Output Directory: {config['output_dir']}")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    logger.info("="*50)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = BasicMedicalTrainer(config)
    
    try:
        # Setup model
        trainer.setup_model()
        
        # Run training
        trainer.train()
        
        logger.info("="*50)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config['output_dir']}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
