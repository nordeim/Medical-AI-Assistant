"""
Model size reduction techniques including pruning and distillation awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import copy
import json

from .config import ReductionConfig, ReductionType


logger = logging.getLogger(__name__)


class PruningMethod(Enum):
    """Methods for neural network pruning."""
    MAGNITUDE = "magnitude"          # Weight magnitude pruning
    GRADIENT = "gradient"            # Gradient-based pruning
    ACTIVATION = "activation"        # Activation-based pruning
    STRUCTURED = "structured"        # Structured pruning (channels/neurons)
    UNSTRUCTURED = "unstructured"    # Unstructured pruning (individual weights)
    GRADUAL = "gradual"              # Gradual pruning schedule


class DistillationMethod(Enum):
    """Knowledge distillation methods."""
    SOFT_LABELS = "soft_labels"      # Soft target distillation
    FEATURE_MAP = "feature_map"      # Feature map distillation
    ATTENTION = "attention"          # Attention-based distillation
    PROGRESSIVE = "progressive"      # Progressive distillation
    SEQUENTIAL = "sequential"        # Sequential distillation


@dataclass
class PruningResult:
    """Result of pruning operation."""
    success: bool
    original_parameters: int
    pruned_parameters: int
    compression_ratio: float
    sparsity_ratio: float
    accuracy_preserved: float
    performance_impact_ms: float
    memory_saved_mb: float
    error_message: Optional[str] = None
    pruning_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationResult:
    """Result of knowledge distillation."""
    success: bool
    original_model_size_mb: float
    student_model_size_mb: float
    compression_ratio: float
    accuracy_retention: float
    performance_improvement: float
    distillation_time_hours: float
    teacher_accuracy: float
    student_accuracy: float
    error_message: Optional[str] = None
    distillation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReductionMetrics:
    """Metrics for model reduction operations."""
    total_reduction_applied: float
    parameters_removed: int
    memory_saved_mb: float
    speedup_factor: float
    accuracy_change: float
    reduction_methods: List[str]
    validation_results: Dict[str, float]


class ModelReducer:
    """
    Advanced model reduction framework supporting multiple pruning and distillation techniques.
    Designed for medical models with high accuracy preservation requirements.
    """
    
    def __init__(self, config: ReductionConfig):
        self.config = config
        self.original_model = None
        self.current_model = None
        
        # Tracking
        self.reduction_history = []
        self.validation_cache = {}
        
        # Medical model considerations
        self.medical_critical_layers = []
        self.accuracy_threshold = 0.95  # Minimum acceptable accuracy for medical models
        
        logger.info("Model reducer initialized")
    
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model structure and identify reduction opportunities."""
        analysis = {
            "model_type": type(model).__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "model_size_mb": self._calculate_model_size(model),
            "layer_analysis": self._analyze_layers(model),
            "reduction_opportunities": self._identify_reduction_opportunities(model),
            "medical_considerations": self._analyze_medical_requirements(model)
        }
        
        # Add depth and complexity analysis
        analysis.update({
            "depth": self._calculate_model_depth(model),
            "complexity_score": self._calculate_complexity_score(model),
            "parameter_distribution": self._analyze_parameter_distribution(model)
        })
        
        logger.info(f"Model analysis complete: {analysis['total_parameters']} parameters, "
                   f"{analysis['model_size_mb']:.1f}MB")
        
        return analysis
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (total_size + buffer_size) / (1024**2)
    
    def _analyze_layers(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model layers for reduction potential."""
        layer_analysis = {
            "linear_layers": [],
            "convolution_layers": [],
            "attention_layers": [],
            "embedding_layers": [],
            "normalization_layers": []
        }
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_analysis["linear_layers"].append({
                    "name": name,
                    "input_features": module.in_features,
                    "output_features": module.out_features,
                    "parameters": module.weight.numel(),
                    "reducible": True
                })
            elif isinstance(module, nn.Conv1d):
                layer_analysis["convolution_layers"].append({
                    "name": name,
                    "input_channels": module.in_channels,
                    "output_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "parameters": module.weight.numel(),
                    "reducible": True
                })
            elif "attention" in name.lower():
                layer_analysis["attention_layers"].append({
                    "name": name,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "reducible": True  # Attention mechanisms are often compressible
                })
            elif isinstance(module, nn.Embedding):
                layer_analysis["embedding_layers"].append({
                    "name": name,
                    "num_embeddings": module.num_embeddings,
                    "embedding_dim": module.embedding_dim,
                    "parameters": module.weight.numel(),
                    "reducible": True
                })
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                layer_analysis["normalization_layers"].append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "reducible": False  # Keep normalization layers
                })
        
        return layer_analysis
    
    def _identify_reduction_opportunities(self, model: nn.Module) -> Dict[str, Any]:
        """Identify specific reduction opportunities."""
        opportunities = {
            "pruning_candidates": [],
            "distillation_candidates": [],
            "quantization_candidates": [],
            "layer_fusion_candidates": []
        }
        
        # Identify pruning candidates
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
                opportunities["pruning_candidates"].append({
                    "layer": name,
                    "type": "linear",
                    "parameters": module.weight.numel(),
                    "sparsity_potential": self._estimate_sparsity_potential(module)
                })
            elif "attention" in name.lower():
                opportunities["distillation_candidates"].append({
                    "layer": name,
                    "type": "attention",
                    "parameters": sum(p.numel() for p in module.parameters()),
                    "distillation_potential": "high"
                })
        
        return opportunities
    
    def _estimate_sparsity_potential(self, module: nn.Module) -> float:
        """Estimate potential for weight sparsity."""
        if isinstance(module, nn.Linear):
            # Estimate based on weight distribution
            weights = module.weight.data.abs().flatten()
            
            # Calculate how many weights are close to zero
            threshold = torch.quantile(weights, 0.1)  # Bottom 10%
            sparse_potential = (weights < threshold).float().mean().item()
            
            return sparse_potential
        
        return 0.0
    
    def _analyze_medical_requirements(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model for medical-specific considerations."""
        medical_analysis = {
            "critical_layers": [],
            "accuracy_sensitive_modules": [],
            "safety_considerations": [],
            "compliance_flags": []
        }
        
        # Identify medical-critical layers
        for name, module in model.named_modules():
            if self._is_medical_critical_layer(name, module):
                medical_analysis["critical_layers"].append({
                    "layer": name,
                    "type": type(module).__name__,
                    "criticality": "high",
                    "protection_required": True
                })
            
            # Check for accuracy-sensitive modules
            if self._is_accuracy_sensitive_module(name, module):
                medical_analysis["accuracy_sensitive_modules"].append({
                    "layer": name,
                    "sensitivity": "high",
                    "preservation_priority": "maximum"
                })
        
        # Medical safety considerations
        medical_analysis["safety_considerations"] = [
            "Model must maintain clinical accuracy standards",
            "No reduction in critical decision-making layers",
            "Preserve interpretability features",
            "Maintain bias detection capabilities"
        ]
        
        # Compliance flags
        medical_analysis["compliance_flags"] = [
            "HIPAA_compliance_required",
            "FDA_guidelines_consideration",
            "Medical_device_regulations"
        ]
        
        return medical_analysis
    
    def _is_medical_critical_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer is critical for medical applications."""
        critical_keywords = [
            "diagnosis", "classification", "prediction", "decision",
            "clinical", "medical", "safety", "critical"
        ]
        
        name_lower = name.lower()
        
        # Check for critical layer names
        if any(keyword in name_lower for keyword in critical_keywords):
            return True
        
        # Check for output/decision layers
        if isinstance(module, nn.Linear) and "output" in name_lower:
            return True
        
        return False
    
    def _is_accuracy_sensitive_module(self, name: str, module: nn.Module) -> bool:
        """Determine if a module is sensitive to accuracy changes."""
        sensitive_keywords = [
            "classifier", "decoder", "final", "output", "head"
        ]
        
        name_lower = name.lower()
        
        # Check for sensitive module names
        if any(keyword in name_lower for keyword in sensitive_keywords):
            return True
        
        return False
    
    def _calculate_model_depth(self, model: nn.Module) -> int:
        """Calculate model depth (number of layers)."""
        depth = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                depth += 1
        
        return depth
    
    def _calculate_complexity_score(self, model: nn.Module) -> float:
        """Calculate model complexity score."""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Normalize by model type
        if total_params > 1e9:  # >1B parameters
            return 10.0
        elif total_params > 1e8:  # >100M parameters
            return 8.0
        elif total_params > 1e7:  # >10M parameters
            return 6.0
        elif total_params > 1e6:  # >1M parameters
            return 4.0
        else:
            return 2.0
    
    def _analyze_parameter_distribution(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze parameter distribution across layers."""
        layer_sizes = []
        total_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                layer_params = sum(p.numel() for p in module.parameters())
                layer_sizes.append((name, layer_params))
                total_params += layer_params
        
        # Sort by size
        layer_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate distribution statistics
        sizes = [size for _, size in layer_sizes]
        
        return {
            "largest_layer_params": max(sizes) if sizes else 0,
            "smallest_layer_params": min(sizes) if sizes else 0,
            "average_layer_params": np.mean(sizes) if sizes else 0,
            "std_layer_params": np.std(sizes) if sizes else 0,
            "top_5_layers": layer_sizes[:5]
        }
    
    def apply_pruning(self, 
                     model: nn.Module,
                     pruning_ratio: float = None,
                     method: PruningMethod = PruningMethod.MAGNITUDE,
                     validation_function: Optional[Callable] = None) -> PruningResult:
        """
        Apply pruning to reduce model size while preserving accuracy.
        
        Args:
            model: Model to prune
            pruning_ratio: Ratio of weights to prune (0.0-1.0)
            method: Pruning method to use
            validation_function: Function to validate model accuracy
            
        Returns:
            PruningResult with detailed metrics
        """
        if pruning_ratio is None:
            pruning_ratio = self.config.prune_ratio
        
        if pruning_ratio <= 0.0:
            return PruningResult(
                success=True,
                original_parameters=sum(p.numel() for p in model.parameters()),
                pruned_parameters=0,
                compression_ratio=1.0,
                sparsity_ratio=0.0,
                accuracy_preserved=1.0,
                performance_impact_ms=0.0,
                memory_saved_mb=0.0,
                pruning_details={"message": "No pruning requested"}
            )
        
        start_time = time.time()
        original_params = sum(p.numel() for p in model.parameters())
        original_size_mb = self._calculate_model_size(model)
        
        try:
            # Create a copy for pruning
            model_copy = copy.deepcopy(model)
            
            # Apply pruning based on method
            if method == PruningMethod.MAGNITUDE:
                pruned_model = self._apply_magnitude_pruning(model_copy, pruning_ratio)
            elif method == PruningMethod.STRUCTURED:
                pruned_model = self._apply_structured_pruning(model_copy, pruning_ratio)
            elif method == PruningMethod.GRADUAL:
                pruned_model = self._apply_gradual_pruning(model_copy, pruning_ratio)
            else:
                pruned_model = self._apply_magnitude_pruning(model_copy, pruning_ratio)
            
            # Validate accuracy if validation function provided
            accuracy_preserved = 1.0
            if validation_function:
                try:
                    accuracy_preserved = validation_function(pruned_model)
                    logger.info(f"Post-pruning accuracy: {accuracy_preserved:.3f}")
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")
                    accuracy_preserved = 0.0
            
            # Check medical accuracy requirements
            if accuracy_preserved < self.accuracy_threshold:
                raise ValueError(f"Pruning would reduce accuracy below medical threshold: {accuracy_preserved:.3f} < {self.accuracy_threshold}")
            
            # Update original model
            self.original_model = model
            self.current_model = pruned_model
            
            # Calculate results
            pruned_params = sum(p.numel() for p in pruned_model.parameters())
            pruned_size_mb = self._calculate_model_size(pruned_model)
            
            compression_ratio = original_params / pruned_params if pruned_params > 0 else 1.0
            sparsity_ratio = 1.0 - (pruned_params / original_params)
            memory_saved_mb = original_size_mb - pruned_size_mb
            performance_impact = (time.time() - start_time) * 1000
            
            # Copy pruned weights back to original model
            self._copy_pruned_weights(model, pruned_model)
            
            result = PruningResult(
                success=True,
                original_parameters=original_params,
                pruned_parameters=pruned_params,
                compression_ratio=compression_ratio,
                sparsity_ratio=sparsity_ratio,
                accuracy_preserved=accuracy_preserved,
                performance_impact_ms=performance_impact,
                memory_saved_mb=memory_saved_mb,
                pruning_details={
                    "method": method.value,
                    "pruning_ratio": pruning_ratio,
                    "original_size_mb": original_size_mb,
                    "pruned_size_mb": pruned_size_mb
                }
            )
            
            logger.info(f"Pruning completed: {compression_ratio:.1f}x compression, "
                       f"{sparsity_ratio:.1%} sparsity, {accuracy_preserved:.3f} accuracy preserved")
            
            return result
            
        except Exception as e:
            error_msg = f"Pruning failed: {str(e)}"
            logger.error(error_msg)
            
            return PruningResult(
                success=False,
                original_parameters=original_params,
                pruned_parameters=0,
                compression_ratio=1.0,
                sparsity_ratio=0.0,
                accuracy_preserved=0.0,
                performance_impact_ms=0.0,
                memory_saved_mb=0.0,
                error_message=error_msg
            )
    
    def _apply_magnitude_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Apply magnitude-based weight pruning."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get weight matrix
                weight = module.weight.data
                
                # Calculate pruning threshold
                threshold = torch.quantile(weight.abs(), pruning_ratio)
                
                # Create pruning mask
                mask = (weight.abs() > threshold).float()
                
                # Apply mask
                module.weight.data *= mask
                
                # Update bias if exists
                if module.bias is not None:
                    bias = module.bias.data
                    bias_mask = mask.sum(dim=1) > 0  # Keep bias if any input weight kept
                    module.bias.data *= bias_mask.float()
        
        return model
    
    def _apply_structured_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Apply structured pruning (remove entire neurons/channels)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Calculate importance scores for each neuron
                importance_scores = torch.norm(module.weight.data, dim=1)
                
                # Determine how many neurons to prune
                num_neurons = module.out_features
                neurons_to_prune = int(num_neurons * pruning_ratio)
                
                # Find least important neurons
                _, indices = torch.topk(importance_scores, neurons_to_prune, largest=False)
                
                # Create new module with fewer neurons
                new_module = nn.Linear(module.in_features, num_neurons - neurons_to_prune)
                
                # Copy weights of retained neurons
                retained_indices = torch.setdiff1d(torch.arange(num_neurons), indices)
                new_module.weight.data = module.weight.data[retained_indices]
                
                if module.bias is not None:
                    new_module.bias.data = module.bias.data[retained_indices]
                
                # Replace module
                self._replace_module(model, name, new_module)
        
        return model
    
    def _apply_gradual_pruning(self, model: nn.Module, target_ratio: float) -> nn.Module:
        """Apply gradual pruning with multiple iterations."""
        current_ratio = 0.0
        iterations = 10
        increment = target_ratio / iterations
        
        for i in range(iterations):
            current_ratio += increment
            model = self._apply_magnitude_pruning(model, increment)
            
            # Optional: Fine-tune after each iteration
            logger.debug(f"Gradual pruning iteration {i+1}/{iterations}: {current_ratio:.3f}")
        
        return model
    
    def _copy_pruned_weights(self, target_model: nn.Module, source_model: nn.Module):
        """Copy pruned weights from source to target model."""
        for (target_name, target_module), (source_name, source_module) in zip(
            target_model.named_modules(), source_model.named_modules()
        ):
            if isinstance(target_module, nn.Linear) and isinstance(source_module, nn.Linear):
                if target_module.weight.shape == source_module.weight.shape:
                    target_module.weight.data.copy_(source_module.weight.data)
                    if target_module.bias is not None and source_module.bias is not None:
                        if target_module.bias.shape == source_module.bias.shape:
                            target_module.bias.data.copy_(source_module.bias.data)
    
    def _replace_module(self, model: nn.Module, module_path: str, new_module: nn.Module):
        """Replace a module in the model."""
        path_parts = module_path.split('.')
        current = model
        
        # Navigate to parent of target module
        for part in path_parts[:-1]:
            current = getattr(current, part)
        
        # Replace the module
        setattr(current, path_parts[-1], new_module)
    
    def apply_distillation(self,
                          teacher_model: nn.Module,
                          student_model: nn.Module,
                          distillation_data: Optional[torch.utils.data.DataLoader] = None,
                          validation_function: Optional[Callable] = None) -> DistillationResult:
        """
        Apply knowledge distillation to create a smaller student model.
        
        Args:
            teacher_model: Larger model to distill knowledge from
            student_model: Smaller model to train
            distillation_data: Data for distillation training
            validation_function: Function to validate model accuracy
            
        Returns:
            DistillationResult with detailed metrics
        """
        start_time = time.time()
        
        try:
            # Get baseline metrics
            teacher_size_mb = self._calculate_model_size(teacher_model)
            student_size_mb = self._calculate_model_size(student_model)
            
            # Validate teacher accuracy if validation function provided
            teacher_accuracy = 1.0
            if validation_function:
                try:
                    teacher_accuracy = validation_function(teacher_model)
                    logger.info(f"Teacher accuracy: {teacher_accuracy:.3f}")
                except Exception as e:
                    logger.warning(f"Teacher validation failed: {e}")
            
            # Apply distillation based on method
            if self.config.distill_temperature > 1.0:
                distilled_student = self._apply_soft_label_distillation(
                    teacher_model, student_model, distillation_data
                )
            else:
                distilled_student = self._apply_feature_distillation(
                    teacher_model, student_model, distillation_data
                )
            
            # Validate student accuracy
            student_accuracy = teacher_accuracy  # Placeholder
            if validation_function:
                try:
                    student_accuracy = validation_function(distilled_student)
                    logger.info(f"Student accuracy after distillation: {student_accuracy:.3f}")
                except Exception as e:
                    logger.warning(f"Student validation failed: {e}")
                    student_accuracy = 0.0
            
            # Calculate results
            compression_ratio = teacher_size_mb / student_size_mb
            accuracy_retention = student_accuracy / teacher_accuracy if teacher_accuracy > 0 else 0.0
            distillation_time = (time.time() - start_time) / 3600  # Convert to hours
            
            result = DistillationResult(
                success=True,
                original_model_size_mb=teacher_size_mb,
                student_model_size_mb=student_size_mb,
                compression_ratio=compression_ratio,
                accuracy_retention=accuracy_retention,
                performance_improvement=student_size_mb / teacher_size_mb,
                distillation_time_hours=distillation_time,
                teacher_accuracy=teacher_accuracy,
                student_accuracy=student_accuracy,
                distillation_details={
                    "temperature": self.config.distill_temperature,
                    "alpha": self.config.distill_alpha,
                    "method": "soft_labels" if self.config.distill_temperature > 1.0 else "feature_map"
                }
            )
            
            logger.info(f"Distillation completed: {compression_ratio:.1f}x compression, "
                       f"{accuracy_retention:.3f} accuracy retention")
            
            return result
            
        except Exception as e:
            error_msg = f"Distillation failed: {str(e)}"
            logger.error(error_msg)
            
            return DistillationResult(
                success=False,
                original_model_size_mb=self._calculate_model_size(teacher_model),
                student_model_size_mb=self._calculate_model_size(student_model),
                compression_ratio=1.0,
                accuracy_retention=0.0,
                performance_improvement=1.0,
                distillation_time_hours=0.0,
                teacher_accuracy=0.0,
                student_accuracy=0.0,
                error_message=error_msg
            )
    
    def _apply_soft_label_distillation(self,
                                     teacher_model: nn.Module,
                                     student_model: nn.Module,
                                     distillation_data: Optional[torch.utils.data.DataLoader]) -> nn.Module:
        """Apply soft label knowledge distillation."""
        if distillation_data is None:
            logger.warning("No distillation data provided, returning student model unchanged")
            return student_model
        
        # Simplified distillation training
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        temperature = self.config.distill_temperature
        alpha = self.config.distill_alpha
        
        for epoch in range(5):  # Simplified training
            for batch_idx, batch in enumerate(distillation_data):
                if batch_idx >= 10:  # Limit for demo
                    break
                
                optimizer.zero_grad()
                
                # Get inputs and targets
                if isinstance(batch, tuple):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                # Forward pass
                student_outputs = student_model(inputs)
                teacher_outputs = teacher_model(inputs)
                
                # Calculate distillation loss
                if targets is not None:
                    # Combined loss with hard and soft targets
                    hard_loss = F.cross_entropy(student_outputs, targets)
                    soft_loss = F.kl_div(
                        F.log_softmax(student_outputs / temperature, dim=1),
                        F.softmax(teacher_outputs / temperature, dim=1),
                        reduction='batchmean'
                    ) * (temperature ** 2)
                    
                    loss = alpha * hard_loss + (1 - alpha) * soft_loss
                else:
                    # Pure distillation loss
                    soft_loss = F.kl_div(
                        F.log_softmax(student_outputs / temperature, dim=1),
                        F.softmax(teacher_outputs / temperature, dim=1),
                        reduction='batchmean'
                    ) * (temperature ** 2)
                    loss = soft_loss
                
                loss.backward()
                optimizer.step()
        
        student_model.eval()
        return student_model
    
    def _apply_feature_distillation(self,
                                   teacher_model: nn.Module,
                                   student_model: nn.Module,
                                   distillation_data: Optional[torch.utils.data.DataLoader]) -> nn.Module:
        """Apply feature map-based knowledge distillation."""
        # This is a simplified feature distillation
        # In practice, you'd extract intermediate representations and match them
        
        if distillation_data is None:
            logger.warning("No distillation data provided, returning student model unchanged")
            return student_model
        
        # For now, just return the student model with some fine-tuning
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(3):  # Simplified training
            for batch_idx, batch in enumerate(distillation_data):
                if batch_idx >= 5:  # Limit for demo
                    break
                
                optimizer.zero_grad()
                
                if isinstance(batch, tuple):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = None
                
                student_outputs = student_model(inputs)
                
                if targets is not None:
                    loss = F.cross_entropy(student_outputs, targets)
                    loss.backward()
                    optimizer.step()
        
        student_model.eval()
        return student_model
    
    def get_reduction_recommendations(self, model_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for model reduction."""
        recommendations = []
        
        total_params = model_analysis["total_parameters"]
        model_size_mb = model_analysis["model_size_mb"]
        
        # Pruning recommendations
        if total_params > 1e7:  # >10M parameters
            if self.config.prune_ratio > 0:
                recommendations.append({
                    "method": "pruning",
                    "type": "magnitude",
                    "description": f"Apply magnitude-based pruning to remove {self.config.prune_ratio:.1%} of weights",
                    "expected_compression": 1 / (1 - self.config.prune_ratio),
                    "risk_level": "medium",
                    "medical_compatible": True
                })
            
            if model_analysis["reduction_opportunities"]["pruning_candidates"]:
                recommendations.append({
                    "method": "structured_pruning",
                    "description": "Apply structured pruning to remove entire neurons/channels",
                    "target_layers": [candidate["layer"] for candidate in model_analysis["reduction_opportunities"]["pruning_candidates"][:5]],
                    "expected_compression": 1.5,
                    "risk_level": "high",
                    "medical_compatible": False  # Higher risk for medical models
                })
        
        # Distillation recommendations
        if total_params > 5e7:  # >50M parameters
            student_size_ratio = self.config.distill_student_size_ratio
            recommendations.append({
                "method": "knowledge_distillation",
                "description": f"Create student model at {student_size_ratio:.1%} of original size",
                "expected_compression": 1 / student_size_ratio,
                "risk_level": "medium",
                "medical_compatible": True,
                "validation_required": True
            })
        
        # Quantization recommendations
        if model_size_mb > 100:  # >100MB model
            recommendations.append({
                "method": "quantization",
                "description": "Apply INT8 or INT4 quantization for memory reduction",
                "expected_compression": 4,  # INT8 typically gives 4x compression
                "risk_level": "low",
                "medical_compatible": True
            })
        
        # Combined approach recommendations
        if total_params > 1e8:  # >100M parameters
            recommendations.append({
                "method": "combined_reduction",
                "description": "Combine pruning (30%) + quantization (INT8) + distillation (50% size)",
                "expected_compression": 10,
                "risk_level": "high",
                "medical_compatible": False,
                "requires_validation": True
            })
        
        return recommendations
    
    def validate_reduction(self, 
                          original_model: nn.Module,
                          reduced_model: nn.Module,
                          validation_dataset: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, Any]:
        """Validate that reduction maintains acceptable performance."""
        validation_results = {
            "size_reduction": {
                "original_size_mb": self._calculate_model_size(original_model),
                "reduced_size_mb": self._calculate_model_size(reduced_model),
                "compression_ratio": self._calculate_model_size(original_model) / self._calculate_model_size(reduced_model)
            },
            "parameter_reduction": {
                "original_parameters": sum(p.numel() for p in original_model.parameters()),
                "reduced_parameters": sum(p.numel() for p in reduced_model.parameters()),
                "parameter_reduction_ratio": 1 - (sum(p.numel() for p in reduced_model.parameters()) / 
                                                  sum(p.numel() for p in original_model.parameters()))
            },
            "performance_validation": {
                "validated": False,
                "accuracy_maintained": False,
                "medical_standards_met": False
            }
        }
        
        # Placeholder for actual validation
        # In practice, you would:
        # 1. Run both models on validation dataset
        # 2. Compare accuracy metrics
        # 3. Check medical-specific requirements
        # 4. Validate inference speed improvements
        
        logger.info("Model reduction validation completed (placeholder implementation)")
        
        return validation_results
    
    def get_reduction_summary(self) -> Dict[str, Any]:
        """Get summary of all reduction operations performed."""
        return {
            "total_reductions": len(self.reduction_history),
            "reduction_history": self.reduction_history,
            "current_model_size_mb": (
                self._calculate_model_size(self.current_model) if self.current_model else 0.0
            ),
            "original_model_size_mb": (
                self._calculate_model_size(self.original_model) if self.original_model else 0.0
            ),
            "total_compression_achieved": (
                self._calculate_model_size(self.original_model) / self._calculate_model_size(self.current_model)
                if self.original_model and self.current_model else 1.0
            )
        }