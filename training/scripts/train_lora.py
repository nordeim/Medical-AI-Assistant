#!/usr/bin/env python3
"""
PEFT/LoRA Training Script with Model Optimization
Comprehensive training script for fine-tuning large language models using LoRA/PEFT.
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

# PEFT and transformers imports
from peft import (
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    AdaLoraConfig,
    IA3Config
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset, load_from_disk

# DeepSpeed and other optimization libraries
try:
    import deepspeed
    from deepspeed import deepspeed_config
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# WandB and other monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for base model setup."""
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        help="Name or path of the base model"
    )
    model_type: str = field(
        default="llama",
        choices=["llama", "mistral", "qwen", "gpt2", "gpt-neo", "opt"],
        help="Type of base model"
    )
    trust_remote_code: bool = field(default=False, help="Trust remote code")
    use_auth_token: Optional[str] = field(default=None, help="HuggingFace token")
    cache_dir: Optional[str] = field(default=None, help="Cache directory for models")
    max_length: int = field(default=2048, help="Maximum sequence length")
    device_map: Optional[str] = field(default=None, help="Device mapping strategy")

@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters."""
    r: int = field(default=16, help="LoRA rank")
    lora_alpha: int = field(default=32, help="LoRA alpha parameter")
    lora_dropout: float = field(default=0.1, help="LoRA dropout rate")
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Target modules for LoRA"
    )
    bias: str = field(default="none", choices=["none", "all", "lora_only"])
    task_type: str = field(default="CAUSAL_LM", help="Task type")
    use_rslora: bool = field(default=False, help="Use RSLoRA")
    use_dora: bool = field(default=False, help="Use DoRA")
    init_lora_weights: bool = field(default=True, help="Initialize LoRA weights")

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    use_4bit: bool = field(default=False, help="Use 4-bit quantization")
    use_8bit: bool = field(default=False, help="Use 8-bit quantization")
    bnb_4bit_quant_type: str = field(default="nf4", help="4-bit quantization type")
    bnb_4bit_use_double_quant: bool = field(default=True, help="Use double quantization")
    bnb_4bit_compute_dtype: torch.dtype = field(
        default=torch.bfloat16,
        help="4-bit compute dtype"
    )
    load_in_4bit: bool = field(default=False, help="Load model in 4-bit")
    load_in_8bit: bool = field(default=False, help="Load model in 8-bit")

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    output_dir: str = field(default="./output", help="Output directory")
    logging_dir: Optional[str] = field(default=None, help="Logging directory")
    num_epochs: int = field(default=3, help="Number of training epochs")
    per_device_train_batch_size: int = field(default=4, help="Training batch size per device")
    per_device_eval_batch_size: int = field(default=4, help="Evaluation batch size per device")
    gradient_accumulation_steps: int = field(default=1, help="Gradient accumulation steps")
    learning_rate: float = field(default=2e-4, help="Learning rate")
    weight_decay: float = field(default=0.01, help="Weight decay")
    max_grad_norm: float = field(default=1.0, help="Maximum gradient norm")
    warmup_ratio: float = field(default=0.1, help="Warmup ratio")
    lr_scheduler_type: str = field(
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"],
        help="Learning rate scheduler type"
    )
    warmup_steps: int = field(default=0, help="Number of warmup steps")
    fp16: bool = field(default=False, help="Use FP16 mixed precision")
    bf16: bool = field(default=False, help="Use BF16 mixed precision")
    dataloader_num_workers: int = field(default=0, help="Number of DataLoader workers")
    remove_unused_columns: bool = field(default=False, help="Remove unused columns")
    save_steps: int = field(default=500, help="Save steps")
    save_total_limit: int = field(default=3, help="Total number of saves to keep")
    evaluation_strategy: str = field(
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy"
    )
    eval_steps: int = field(default=500, help="Evaluation steps")
    load_best_model_at_end: bool = field(default=True, help="Load best model at end")
    metric_for_best_model: str = field(default="eval_loss", help="Metric for best model")
    greater_is_better: bool = field(default=False, help="Greater is better metric")
    early_stopping_patience: int = field(default=3, help="Early stopping patience")
    save_safetensors: bool = field(default=True, help="Save in safetensors format")
    report_to: List[str] = field(
        default_factory=lambda: [],
        help="Report to logging services"
    )
    run_name: Optional[str] = field(default=None, help="Run name for logging")

@dataclass
class OptimizationConfig:
    """Configuration for optimization features."""
    gradient_checkpointing: bool = field(default=True, help="Enable gradient checkpointing")
    group_by_length: bool = field(default=False, help="Group by length")
    length_column_name: str = field(default="length", help="Length column name")
    ddp_find_unused_parameters: bool = field(default=None, help="DDP find unused parameters")
    dataloader_pin_memory: bool = field(default=True, help="Pin memory for DataLoader")
    skip_memory_metrics: bool = field(default=True, skip_memory_metrics=True)
    max_steps: Optional[int] = field(default=None, help="Maximum training steps")
    save_strategy: str = field(
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Save strategy"
    )
    deepspeed_config: Optional[str] = field(default=None, help="DeepSpeed config file")
    use_distributed: bool = field(default=False, help="Use distributed training")
    local_rank: int = field(default=-1, help="Local rank for distributed training")
    gradient_checkpointing_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        help="Additional kwargs for gradient checkpointing"
    )

@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_path: str = field(help="Path to training data")
    data_type: str = field(
        default="jsonl",
        choices=["jsonl", "json", "csv", "txt", "datasets"],
        help="Type of data file"
    )
    text_column: str = field(default="text", help="Text column name")
    prompt_column: Optional[str] = field(default=None, help="Prompt column name")
    response_column: Optional[str] = field(default=None, help="Response column name")
    train_split_ratio: float = field(default=0.9, help="Training split ratio")
    val_split_ratio: float = field(default=0.05, help="Validation split ratio")
    max_samples: Optional[int] = field(default=None, help="Maximum samples to use")
    stream: bool = field(default=False, help="Stream data loading")
    preprocessing_num_workers: int = field(default=None, help="Preprocessing workers")
    load_from_cache_file: bool = field(default=True, help="Load from cache file")
    dataset_cache_dir: Optional[str] = field(default=None, help="Dataset cache directory")

class CustomDataCollator:
    """Custom data collator for dynamic padding and truncation."""
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for data loading."""
        batch = {}
        
        # Handle different input formats
        if "input_ids" in features[0]:
            input_ids = [f["input_ids"] for f in features]
            attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        else:
            # Handle raw text input
            input_ids = [self.tokenizer.encode(
                f.get("text", ""),
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding
            ) for f in features]
            attention_mask = [self.tokenizer.encode(
                f.get("text", ""),
                add_special_tokens=False,
                max_length=self.max_length,
                truncation=self.truncation,
                padding="do_not_pad"
            )[1] for f in features]
        
        # Pad input sequences
        max_length = min(max(len(seq) for seq in input_ids), self.max_length)
        
        batch["input_ids"] = torch.tensor([
            seq[:max_length] + [self.tokenizer.pad_token_id] * (max_length - len(seq))
            for seq in input_ids
        ], dtype=torch.long)
        
        batch["attention_mask"] = torch.tensor([
            mask[:max_length] + [0] * (max_length - len(mask))
            for mask in attention_mask
        ], dtype=torch.long)
        
        # Add labels (shifted input_ids for causal LM)
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        else:
            batch["labels"] = batch["input_ids"].clone()
            # Set labels to -100 for padding tokens
            batch["labels"][batch["attention_mask"] == 0] = -100
        
        return batch

class MemoryMonitoringCallback(TrainerCallback):
    """Custom callback for monitoring memory usage."""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Monitor memory usage at step end."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {cached:.2f}GB")
        return control

class ModelValidator:
    """Utility class for model validation and conversion."""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> None:
        """Validate model configuration."""
        if config.model_type not in ["llama", "mistral", "qwen", "gpt2", "gpt-neo", "opt"]:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        if config.use_auth_token and not config.use_auth_token.startswith("hf_"):
            logger.warning("Auth token doesn't seem to be a valid HuggingFace token")
    
    @staticmethod
    def validate_lora_config(config: LoRAConfig) -> None:
        """Validate LoRA configuration."""
        if config.r <= 0:
            raise ValueError("LoRA rank (r) must be positive")
        
        if config.lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        
        if not (0 <= config.lora_dropout <= 1):
            raise ValueError("LoRA dropout must be between 0 and 1")
        
        valid_bias_values = ["none", "all", "lora_only"]
        if config.bias not in valid_bias_values:
            raise ValueError(f"Invalid bias value. Must be one of {valid_bias_values}")
    
    @staticmethod
    def validate_quantization_config(config: QuantizationConfig) -> None:
        """Validate quantization configuration."""
        if config.use_4bit and config.use_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        if config.load_in_4bit and config.load_in_8bit:
            raise ValueError("Cannot load model in both 4-bit and 8-bit")

class LoRATrainer:
    """Main LoRA training class with comprehensive features."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        quantization_config: QuantizationConfig,
        training_config: TrainingConfig,
        optimization_config: OptimizationConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quantization_config = quantization_config
        self.training_config = training_config
        self.optimization_config = optimization_config
        self.data_config = data_config
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        self.scaler = None
        
        # Validate configurations
        self._validate_configurations()
        
        # Initialize wandb if enabled
        self._initialize_wandb()
    
    def _validate_configurations(self) -> None:
        """Validate all configurations."""
        ModelValidator.validate_model_config(self.model_config)
        ModelValidator.validate_lora_config(self.lora_config)
        ModelValidator.validate_quantization_config(self.quantization_config)
        
        logger.info("All configurations validated successfully")
    
    def _initialize_wandb(self) -> None:
        """Initialize WandB logging if enabled."""
        if "wandb" in self.training_config.report_to and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="lora-training",
                    name=self.training_config.run_name,
                    config={
                        "model_config": self.model_config.__dict__,
                        "lora_config": self.lora_config.__dict__,
                        "quantization_config": self.quantization_config.__dict__,
                        "training_config": self.training_config.__dict__,
                        "optimization_config": self.optimization_config.__dict__,
                        "data_config": self.data_config.__dict__
                    }
                )
                logger.info("WandB initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
    
    def _load_and_prepare_tokenizer(self) -> None:
        """Load and prepare tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code,
            use_auth_token=self.model_config.use_auth_token,
            cache_dir=self.model_config.cache_dir
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side
        self.tokenizer.padding_side = "right"
        
        logger.info("Tokenizer loaded and prepared successfully")
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration."""
        if self.quantization_config.use_4bit or self.quantization_config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quantization_config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.quantization_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self.quantization_config.bnb_4bit_compute_dtype
            )
        elif self.quantization_config.use_8bit or self.quantization_config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None
    
    def _load_and_prepare_model(self) -> None:
        """Load and prepare model with quantization and LoRA."""
        logger.info(f"Loading model: {self.model_config.model_name}")
        
        # Load quantization config
        quantization_config = self._get_quantization_config()
        
        # Load model configuration
        model_kwargs = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "use_auth_token": self.model_config.use_auth_token,
            "cache_dir": self.model_config.cache_dir,
            "torch_dtype": torch.bfloat16 if self.training_config.bf16 else torch.float16,
        }
        
        # Add device map if specified
        if self.model_config.device_map:
            model_kwargs["device_map"] = self.model_config.device_map
        
        # Add quantization config if specified
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            **model_kwargs
        )
        
        # Prepare model for training
        self._prepare_model_for_training()
        
        # Add LoRA adapters
        self._add_lora_adapters()
        
        logger.info("Model loaded and prepared successfully")
    
    def _prepare_model_for_training(self) -> None:
        """Prepare model for training with optimizations."""
        # Enable gradient checkpointing for memory efficiency
        if self.optimization_config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable(**self.optimization_config.gradient_checkpointing_kwargs)
            else:
                self.model.config.use_cache = False
        
        # Prepare for quantization if using
        if self.quantization_config.use_4bit or self.quantization_config.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Set to training mode
        self.model.train()
    
    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to the model."""
        logger.info("Adding LoRA adapters...")
        
        # Create LoRA configuration
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=self.lora_config.use_rslora,
            use_dora=self.lora_config.use_dora
        )
        
        # Add LoRA to model
        self.model = get_peft_model(self.model, peft_config)
        
        logger.info("LoRA adapters added successfully")
    
    def _load_and_prepare_data(self) -> None:
        """Load and prepare training data."""
        logger.info("Loading training data...")
        
        # Load dataset
        if self.data_config.data_type == "datasets":
            dataset = load_dataset(self.data_config.data_path, **self.data_config.get_dataset_kwargs())
        else:
            dataset = load_dataset(self.data_config.data_path, data_files=self.data_config.data_path)
        
        # Convert to Dataset object if needed
        if not isinstance(dataset, Dataset):
            if "train" in dataset:
                dataset = dataset["train"]
            elif isinstance(dataset, dict):
                dataset = Dataset.from_list(dataset["train"] if "train" in dataset else dataset)
        
        # Sample data if max_samples is specified
        if self.data_config.max_samples:
            dataset = dataset.select(range(min(len(dataset), self.data_config.max_samples)))
        
        # Split dataset
        dataset = self._split_dataset(dataset)
        
        # Process dataset
        self.train_dataset, self.eval_dataset = self._process_dataset(dataset)
        
        logger.info(f"Data loaded successfully - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")
    
    def _split_dataset(self, dataset: Dataset) -> Dataset:
        """Split dataset into train, validation, and test sets."""
        splits = dataset.train_test_split(
            test_size=1.0 - self.data_config.train_split_ratio
        )
        
        train_data = splits["train"]
        
        if self.data_config.val_split_ratio > 0:
            remaining_data = splits["test"]
            val_test_splits = remaining_data.train_test_split(
                test_size=self.data_config.val_split_ratio / (self.data_config.val_split_ratio + 0.05)
            )
            eval_data = val_test_splits["train"]
            test_data = val_test_splits["test"]
        else:
            eval_data = splits["test"]
            test_data = None
        
        self.train_dataset = train_data
        self.eval_dataset = eval_data
        
        return train_data
    
    def _process_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Process dataset for training."""
        def tokenize_function(examples):
            """Tokenize examples."""
            texts = []
            
            # Handle different data formats
            if self.data_config.prompt_column and self.data_config.response_column:
                # Structured conversation format
                for prompt, response in zip(
                    examples[self.data_config.prompt_column],
                    examples[self.data_config.response_column]
                ):
                    text = f"{prompt} {response}"
                    texts.append(text)
            else:
                # Simple text format
                texts = examples[self.data_config.text_column]
            
            # Tokenize texts
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.model_config.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Add labels (for causal LM training)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Tokenize datasets
        train_tokenized = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train data",
            num_workers=self.data_config.preprocessing_num_workers,
            load_from_cache_file=self.data_config.load_from_cache_file
        )
        
        eval_tokenized = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
            desc="Tokenizing eval data",
            num_workers=self.data_config.preprocessing_num_workers,
            load_from_cache_file=self.data_config.load_from_cache_file
        )
        
        return train_tokenized, eval_tokenized
    
    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            logging_dir=self.training_config.logging_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            remove_unused_columns=self.training_config.remove_unused_columns,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            save_safetensors=self.training_config.save_safetensors,
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            group_by_length=self.optimization_config.group_by_length,
            length_column_name=self.optimization_config.length_column_name,
            ddp_find_unused_parameters=self.optimization_config.ddp_find_unused_parameters,
            dataloader_pin_memory=self.optimization_config.dataloader_pin_memory,
            skip_memory_metrics=self.optimization_config.skip_memory_metrics,
            max_steps=self.training_config.num_epochs * len(self.train_dataset) // self.training_config.gradient_accumulation_steps,
            save_strategy=self.optimization_config.save_strategy
        )
        
        return training_args
    
    def _setup_distributed_training(self) -> None:
        """Setup distributed training if enabled."""
        if self.optimization_config.use_distributed:
            if not torch.distributed.is_available():
                raise RuntimeError("Distributed training is not available")
            
            torch.cuda.set_device(self.optimization_config.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            
            logger.info(f"Distributed training initialized on rank {self.optimization_config.local_rank}")
    
    def _setup_deepspeed(self, trainer: Trainer) -> Trainer:
        """Setup DeepSpeed if configured."""
        if self.optimization_config.deepspeed_config and DEEPSPEED_AVAILABLE:
            logger.info(f"Loading DeepSpeed configuration: {self.optimization_config.deepspeed_config}")
            
            # Convert to DeepSpeed
            trainer = trainer.to_deepspeed()
            
            logger.info("DeepSpeed setup completed")
        else:
            logger.warning("DeepSpeed not available or not configured")
        
        return trainer
    
    def setup_trainer(self) -> None:
        """Setup the Trainer instance."""
        logger.info("Setting up trainer...")
        
        # Create data collator
        data_collator = CustomDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.model_config.max_length
        )
        
        # Create training arguments
        training_args = self._create_training_arguments()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            callbacks=[
                MemoryMonitoringCallback(),
                EarlyStoppingCallback(early_stopping_patience=self.training_config.early_stopping_patience)
            ] if self.training_config.evaluation_strategy != "no" else [MemoryMonitoringCallback()]
        )
        
        # Setup DeepSpeed if configured
        if self.optimization_config.deepspeed_config and DEEPSPEED_AVAILABLE:
            self.trainer = self._setup_deepspeed(self.trainer)
        
        logger.info("Trainer setup completed")
    
    def train(self) -> None:
        """Run the training process."""
        logger.info("Starting training process...")
        
        try:
            # Setup all components
            self._load_and_prepare_tokenizer()
            self._load_and_prepare_model()
            self._load_and_prepare_data()
            self.setup_trainer()
            
            # Start training
            logger.info("Starting actual training...")
            train_result = self.trainer.train()
            
            # Save the final model
            logger.info("Saving final model...")
            self.trainer.save_model()
            
            # Save training metrics
            if hasattr(train_result, 'metrics'):
                logger.info(f"Training metrics: {train_result.metrics}")
                
                # Log final metrics to wandb if available
                if "wandb" in self.training_config.report_to and WANDB_AVAILABLE:
                    wandb.log(train_result.metrics)
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # Cleanup
            if WANDB_AVAILABLE and wandb.run:
                wandb.finish()
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the trained model."""
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup_trainer() first.")
        
        logger.info("Starting evaluation...")
        
        eval_result = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_result}")
        
        return eval_result
    
    def predict(self, texts: List[str]) -> List[str]:
        """Generate predictions for given texts."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not initialized")
        
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=self.model_config.max_length,
                    truncation=True
                )
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_text = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                predictions.append(generated_text)
        
        return predictions
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the model with LoRA adapters."""
        if path is None:
            path = self.training_config.output_dir
        
        logger.info(f"Saving model to: {path}")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        logger.info("Model saved successfully")
    
    def load_model(self, path: str) -> None:
        """Load a saved model with LoRA adapters."""
        logger.info(f"Loading model from: {path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load base model
        self._load_and_prepare_model()
        
        # Load LoRA adapters
        self.model = PeftModelForCausalLM.from_pretrained(self.model, path)
        
        logger.info("Model loaded successfully")

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    except ImportError:
        logger.error("PyYAML not installed. Please install with: pip install pyyaml")
        raise
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    
    # Configuration arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None, 
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config_dict = load_config_from_yaml(args.config)
        else:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
        
        # Create configuration objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        lora_config = LoRAConfig(**config_dict.get('lora', {}))
        quantization_config = QuantizationConfig(**config_dict.get('quantization', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        optimization_config = OptimizationConfig(**config_dict.get('optimization', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        # Set local rank for distributed training
        optimization_config.local_rank = args.local_rank
        
        # Create trainer
        trainer = LoRATrainer(
            model_config=model_config,
            lora_config=lora_config,
            quantization_config=quantization_config,
            training_config=training_config,
            optimization_config=optimization_config,
            data_config=data_config
        )
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()