"""
Augmentation Strategy Optimization Module for Medical AI Training

This module provides comprehensive optimization capabilities for data augmentation strategies:
- Strategy performance evaluation
- Parameter optimization using various algorithms
- A/B testing framework for augmentation methods
- Real-time strategy adjustment based on quality feedback
- Multi-objective optimization for balanced improvements
"""

import numpy as np
import json
import logging
import time
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import concurrent.futures
from threading import Lock

try:
    from .data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline
    from .data_quality_assessment import DataQualityAssessment, QualityMetrics
except ImportError:
    # Fallback for direct execution
    from data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline
    from data_quality_assessment import DataQualityAssessment, QualityMetrics


@dataclass
class OptimizationConfig:
    """Configuration for augmentation strategy optimization"""
    # Algorithm settings
    algorithm: str = "genetic"  # genetic, grid_search, random, bayesian
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Evaluation settings
    evaluation_metric: str = "overall_score"  # overall_score, safety_score, diversity_score
    validation_split: float = 0.2
    cross_validation_folds: int = 3
    
    # Performance thresholds
    min_quality_threshold: float = 0.7
    max_processing_time: float = 300.0  # 5 minutes
    early_stopping_patience: int = 3
    
    # Multi-objective weights
    semantic_weight: float = 0.3
    medical_weight: float = 0.3
    safety_weight: float = 0.2
    diversity_weight: float = 0.2
    
    # Resource constraints
    max_memory_usage_mb: int = 2048
    max_cpu_cores: int = 4


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_strategy: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_data: Dict[str, List[float]]
    performance_metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StrategyEvaluator:
    """Evaluates augmentation strategy performance"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.quality_assessor = DataQualityAssessment()
        
        # Evaluation cache
        self._cache = {}
        self._cache_lock = Lock()
    
    def evaluate_strategy(self, 
                        strategy_config: Dict[str, Any], 
                        training_data: List[Dict],
                        validation_data: List[Dict] = None) -> Dict[str, float]:
        """Evaluate an augmentation strategy"""
        
        strategy_key = self._get_strategy_key(strategy_config)
        
        # Check cache first
        with self._cache_lock:
            if strategy_key in self._cache:
                self.logger.info(f"Using cached evaluation for strategy: {strategy_key}")
                return self._cache[strategy_key]
        
        start_time = time.time()
        
        try:
            # Create augmentation config
            aug_config = self._create_augmentation_config(strategy_config)
            
            # Create augmentor
            augmentor = DataAugmentor(aug_config)
            
            # Apply augmentation to training data
            augmented_results = apply_augmentation_pipeline(training_data, aug_config, augmentor)
            
            # Evaluate quality of augmented data
            augmented_conversations = augmented_results.get("augmented_conversations", [])
            
            if not augmented_conversations:
                self.logger.warning("No augmented conversations generated")
                return {"score": 0.0, "details": {"error": "No augmented data generated"}}
            
            # Perform quality assessment
            quality_metrics = self.quality_assessor.assess_data_quality(augmented_conversations)
            
            # Calculate performance scores
            scores = self._calculate_performance_scores(quality_metrics)
            
            # Add timing information
            processing_time = time.time() - start_time
            scores["processing_time"] = processing_time
            
            # Cache results
            with self._cache_lock:
                self._cache[strategy_key] = scores
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Strategy evaluation failed: {str(e)}")
            return {
                "score": 0.0, 
                "details": {"error": str(e), "processing_time": time.time() - start_time}
            }
    
    def _get_strategy_key(self, strategy_config: Dict[str, Any]) -> str:
        """Generate cache key for strategy"""
        config_str = json.dumps(strategy_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _create_augmentation_config(self, strategy_config: Dict[str, Any]) -> AugmentationConfig:
        """Create AugmentationConfig from strategy parameters"""
        
        # Default configuration
        config_dict = {
            "synonym_probability": 0.3,
            "paraphrase_probability": 0.4,
            "back_translation_probability": 0.2,
            "masked_lm_probability": 0.15,
            "style_transfer_probability": 0.25,
            "symptom_variation_probability": 0.4,
            "demographic_diversity_probability": 0.3,
            "scenario_augmentation_probability": 0.5,
            "conversation_flow_probability": 0.35,
            "emergency_routine_balance_probability": 0.4,
            "diversity_threshold": 0.8,
            "semantic_similarity_threshold": 0.7,
            "medical_accuracy_threshold": 0.95,
            "safety_constraint_probability": 0.9,
            "coherence_check_probability": 0.8,
            "max_augmentations": 5,
            "preserve_medical_terms": True,
            "context_aware": True,
            "medical_term_preservation_rate": 0.8,
            "emergency_case_target_ratio": 0.3,
            "routine_case_target_ratio": 0.7,
            "age_group_diversity": True,
            "gender_diversity": True,
            "cultural_diversity": True,
            "socioeconomic_diversity": True,
            "enable_quality_checks": True,
            "enable_safety_validation": True,
            "enable_coherence_tracking": True,
            "min_text_length": 10,
            "max_text_length": 1000,
            "batch_size": 100,
            "max_workers": 4,
            "enable_caching": True,
            "cache_dir": ".augmentation_cache"
        }
        
        # Update with strategy parameters
        config_dict.update(strategy_config)
        
        return AugmentationConfig(**config_dict)
    
    def _calculate_performance_scores(self, quality_metrics: QualityMetrics) -> Dict[str, float]:
        """Calculate performance scores based on quality metrics"""
        
        # Individual metric scores
        semantic_score = (
            quality_metrics.semantic_similarity * 0.4 +
            quality_metrics.semantic_consistency * 0.3 +
            quality_metrics.semantic_coherence * 0.3
        )
        
        medical_score = (
            quality_metrics.medical_accuracy * 0.5 +
            quality_metrics.medical_term_usage * 0.2 +
            quality_metrics.symptom_recognition * 0.15 +
            quality_metrics.diagnostic_logic * 0.15
        )
        
        safety_score = (
            quality_metrics.safety_score * 0.6 +
            quality_metrics.phi_protection * 0.4
        )
        
        diversity_score = (
            quality_metrics.vocabulary_diversity * 0.3 +
            quality_metrics.syntactic_diversity * 0.2 +
            quality_metrics.content_diversity * 0.3 +
            quality_metrics.demographic_diversity * 0.2
        )
        
        coherence_score = (
            quality_metrics.conversation_coherence * 0.4 +
            quality_metrics.logical_flow * 0.3 +
            quality_metrics.contextual_relevance * 0.3
        )
        
        # Weighted overall score
        overall_score = (
            semantic_score * self.config.semantic_weight +
            medical_score * self.config.medical_weight +
            safety_score * self.config.safety_weight +
            diversity_score * self.config.diversity_weight
        )
        
        return {
            "overall_score": overall_score,
            "semantic_score": semantic_score,
            "medical_score": medical_score,
            "safety_score": safety_score,
            "diversity_score": diversity_score,
            "coherence_score": coherence_score,
            "quality_metrics": quality_metrics
        }


class GeneticOptimizer:
    """Genetic algorithm for optimization of augmentation strategies"""
    
    def __init__(self, evaluator: StrategyEvaluator, config: OptimizationConfig):
        self.evaluator = evaluator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parameter space definition
        self.parameter_space = {
            "synonym_probability": (0.0, 1.0),
            "paraphrase_probability": (0.0, 1.0),
            "back_translation_probability": (0.0, 0.5),
            "masked_lm_probability": (0.0, 0.3),
            "style_transfer_probability": (0.0, 0.5),
            "symptom_variation_probability": (0.0, 1.0),
            "demographic_diversity_probability": (0.0, 0.8),
            "scenario_augmentation_probability": (0.0, 1.0),
            "conversation_flow_probability": (0.0, 0.8),
            "max_augmentations": (1, 10),
            "diversity_threshold": (0.5, 1.0),
            "semantic_similarity_threshold": (0.5, 0.95),
            "medical_accuracy_threshold": (0.8, 1.0)
        }
    
    def optimize(self, training_data: List[Dict]) -> OptimizationResult:
        """Run genetic optimization"""
        
        self.logger.info("Starting genetic optimization...")
        
        # Initialize population
        population = self._initialize_population()
        
        # Track optimization history
        history = []
        convergence_data = defaultdict(list)
        best_scores = []
        
        for generation in range(self.config.generations):
            self.logger.info(f"Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate population
            evaluated_population = []
            for individual in population:
                score = self.evaluator.evaluate_strategy(individual, training_data)
                evaluated_population.append((individual, score))
            
            # Sort by fitness
            evaluated_population.sort(key=lambda x: x[1].get("overall_score", 0), reverse=True)
            
            # Track best individual
            best_individual, best_score = evaluated_population[0]
            best_scores.append(best_score.get("overall_score", 0))
            
            # Record generation statistics
            generation_stats = {
                "generation": generation,
                "best_score": best_score.get("overall_score", 0),
                "avg_score": np.mean([s.get("overall_score", 0) for _, s in evaluated_population]),
                "best_individual": best_individual,
                "timestamp": datetime.now().isoformat()
            }
            history.append(generation_stats)
            
            # Convergence tracking
            convergence_data["best_scores"].append(best_score.get("overall_score", 0))
            convergence_data["avg_scores"].append(np.mean([s.get("overall_score", 0) for _, s in evaluated_population]))
            
            # Early stopping check
            if self._should_stop_early(best_scores):
                self.logger.info(f"Early stopping at generation {generation}")
                break
            
            # Create next generation
            population = self._create_next_generation(evaluated_population)
        
        # Final evaluation
        final_score = self.evaluator.evaluate_strategy(best_individual, training_data)
        
        result = OptimizationResult(
            best_strategy=best_individual,
            best_score=final_score.get("overall_score", 0),
            optimization_history=history,
            convergence_data=dict(convergence_data),
            performance_metrics=final_score
        )
        
        self.logger.info(f"Genetic optimization complete. Best score: {result.best_score:.4f}")
        return result
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.config.population_size):
            individual = {}
            for param_name, (min_val, max_val) in self.parameter_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    individual[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    individual[param_name] = np.random.uniform(min_val, max_val)
            population.append(individual)
        
        return population
    
    def _create_next_generation(self, evaluated_population: List[Tuple[Dict, Dict]]) -> List[Dict]:
        """Create next generation through selection, crossover, and mutation"""
        
        population_size = len(evaluated_population)
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = max(2, population_size // 10)
        for i in range(elite_count):
            new_population.append(evaluated_population[i][0].copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < population_size:
            # Selection
            parent1 = self._tournament_selection(evaluated_population)
            parent2 = self._tournament_selection(evaluated_population)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:population_size]
    
    def _tournament_selection(self, evaluated_population: List[Tuple[Dict, Dict]], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection"""
        
        tournament = np.random.choice(len(evaluated_population), tournament_size, replace=False)
        tournament = [evaluated_population[i] for i in tournament]
        
        # Return individual with highest score
        tournament.sort(key=lambda x: x[1].get("overall_score", 0), reverse=True)
        return tournament[0][0].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover"""
        
        child1, child2 = {}, {}
        
        for param_name in self.parameter_space.keys():
            if np.random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Gaussian mutation"""
        
        mutated = individual.copy()
        
        for param_name, (min_val, max_val) in self.parameter_space.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer parameter
                    std = (max_val - min_val) * 0.1
                    mutated[param_name] = int(np.clip(
                        individual[param_name] + np.random.normal(0, std),
                        min_val, max_val
                    ))
                else:
                    # Float parameter
                    std = (max_val - min_val) * 0.1
                    mutated[param_name] = np.clip(
                        individual[param_name] + np.random.normal(0, std),
                        min_val, max_val
                    )
        
        return mutated
    
    def _should_stop_early(self, best_scores: List[float], patience: int = None) -> bool:
        """Check if optimization should stop early"""
        
        if len(best_scores) < self.config.early_stopping_patience:
            return False
        
        recent_scores = best_scores[-self.config.early_stopping_patience:]
        
        # Stop if no improvement in recent generations
        if all(score <= recent_scores[0] for score in recent_scores):
            return True
        
        return False


class GridSearchOptimizer:
    """Grid search optimization"""
    
    def __init__(self, evaluator: StrategyEvaluator, config: OptimizationConfig):
        self.evaluator = evaluator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parameter space (simplified for grid search)
        self.parameter_grid = {
            "synonym_probability": [0.1, 0.3, 0.5],
            "paraphrase_probability": [0.2, 0.4, 0.6],
            "max_augmentations": [2, 3, 5],
            "diversity_threshold": [0.7, 0.8, 0.9]
        }
    
    def optimize(self, training_data: List[Dict]) -> OptimizationResult:
        """Run grid search optimization"""
        
        self.logger.info("Starting grid search optimization...")
        
        # Generate parameter combinations
        param_names = list(self.parameter_grid.keys())
        param_values = list(self.parameter_grid.values())
        
        combinations = list(itertools.product(*param_values))
        
        # Add default parameters
        default_params = {
            "back_translation_probability": 0.2,
            "masked_lm_probability": 0.15,
            "style_transfer_probability": 0.25,
            "symptom_variation_probability": 0.4,
            "demographic_diversity_probability": 0.3,
            "scenario_augmentation_probability": 0.5,
            "conversation_flow_probability": 0.35,
            "semantic_similarity_threshold": 0.7,
            "medical_accuracy_threshold": 0.95,
            "preserve_medical_terms": True,
            "context_aware": True,
            "enable_quality_checks": True,
            "enable_safety_validation": True
        }
        
        best_score = 0.0
        best_strategy = {}
        history = []
        
        for i, combination in enumerate(combinations):
            self.logger.info(f"Evaluating combination {i + 1}/{len(combinations)}")
            
            # Create strategy config
            strategy_config = default_params.copy()
            for param_name, param_value in zip(param_names, combination):
                strategy_config[param_name] = param_value
            
            # Evaluate strategy
            score_result = self.evaluator.evaluate_strategy(strategy_config, training_data)
            score = score_result.get("overall_score", 0)
            
            # Record history
            history.append({
                "combination_index": i,
                "strategy": strategy_config,
                "score": score,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_strategy = strategy_config.copy()
        
        result = OptimizationResult(
            best_strategy=best_strategy,
            best_score=best_score,
            optimization_history=history,
            convergence_data={"scores": [h["score"] for h in history]},
            performance_metrics=self.evaluator.evaluate_strategy(best_strategy, training_data)
        )
        
        self.logger.info(f"Grid search complete. Best score: {result.best_score:.4f}")
        return result


class RandomOptimizer:
    """Random search optimization"""
    
    def __init__(self, evaluator: StrategyEvaluator, config: OptimizationConfig):
        self.evaluator = evaluator
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parameter space
        self.parameter_space = {
            "synonym_probability": (0.0, 0.8),
            "paraphrase_probability": (0.0, 0.8),
            "back_translation_probability": (0.0, 0.4),
            "masked_lm_probability": (0.0, 0.3),
            "style_transfer_probability": (0.0, 0.6),
            "max_augmentations": (1, 8),
            "diversity_threshold": (0.6, 1.0),
            "semantic_similarity_threshold": (0.5, 0.95)
        }
    
    def optimize(self, training_data: List[Dict]) -> OptimizationResult:
        """Run random optimization"""
        
        self.logger.info("Starting random optimization...")
        
        best_score = 0.0
        best_strategy = {}
        history = []
        
        for i in range(self.config.population_size):
            self.logger.info(f"Random search iteration {i + 1}/{self.config.population_size}")
            
            # Generate random strategy
            strategy_config = self._generate_random_strategy()
            
            # Evaluate strategy
            score_result = self.evaluator.evaluate_strategy(strategy_config, training_data)
            score = score_result.get("overall_score", 0)
            
            # Record history
            history.append({
                "iteration": i,
                "strategy": strategy_config,
                "score": score,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update best
            if score > best_score:
                best_score = score
                best_strategy = strategy_config.copy()
        
        result = OptimizationResult(
            best_strategy=best_strategy,
            best_score=best_score,
            optimization_history=history,
            convergence_data={"scores": [h["score"] for h in history]},
            performance_metrics=self.evaluator.evaluate_strategy(best_strategy, training_data)
        )
        
        self.logger.info(f"Random optimization complete. Best score: {result.best_score:.4f}")
        return result
    
    def _generate_random_strategy(self) -> Dict[str, Any]:
        """Generate random strategy configuration"""
        
        strategy_config = {
            "back_translation_probability": 0.2,
            "masked_lm_probability": 0.15,
            "style_transfer_probability": 0.25,
            "symptom_variation_probability": 0.4,
            "demographic_diversity_probability": 0.3,
            "scenario_augmentation_probability": 0.5,
            "conversation_flow_probability": 0.35,
            "medical_accuracy_threshold": 0.95,
            "preserve_medical_terms": True,
            "context_aware": True,
            "enable_quality_checks": True,
            "enable_safety_validation": True
        }
        
        # Add random parameters
        for param_name, (min_val, max_val) in self.parameter_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                strategy_config[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                strategy_config[param_name] = np.random.uniform(min_val, max_val)
        
        return strategy_config


class OptimizationOrchestrator:
    """Main optimization orchestrator"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        self.evaluator = StrategyEvaluator(self.config)
        
        # Optimization history
        self.optimization_history = []
        
        # Optimization methods
        self.optimizers = {
            "genetic": lambda data: GeneticOptimizer(self.evaluator, self.config).optimize(data),
            "grid_search": lambda data: GridSearchOptimizer(self.evaluator, self.config).optimize(data),
            "random": lambda data: RandomOptimizer(self.evaluator, self.config).optimize(data)
        }
    
    def optimize_strategy(self, 
                        training_data: List[Dict],
                        algorithm: str = None) -> OptimizationResult:
        """Run optimization using specified or configured algorithm"""
        
        if algorithm is None:
            algorithm = self.config.algorithm
        
        if algorithm not in self.optimizers:
            raise ValueError(f"Unknown optimization algorithm: {algorithm}")
        
        self.logger.info(f"Starting optimization using {algorithm} algorithm...")
        
        start_time = time.time()
        
        try:
            # Run optimization
            result = self.optimizers[algorithm](training_data)
            
            # Add metadata
            result.optimization_time = time.time() - start_time
            result.algorithm_used = algorithm
            result.config_used = self.config
            
            # Store in history
            self.optimization_history.append(result)
            
            # Save result
            self._save_optimization_result(result, algorithm)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def compare_algorithms(self, 
                          training_data: List[Dict],
                          algorithms: List[str] = None) -> Dict[str, OptimizationResult]:
        """Compare multiple optimization algorithms"""
        
        if algorithms is None:
            algorithms = ["genetic", "grid_search", "random"]
        
        self.logger.info(f"Comparing algorithms: {algorithms}")
        
        results = {}
        
        for algorithm in algorithms:
            if algorithm in self.optimizers:
                self.logger.info(f"Running {algorithm} optimization...")
                
                # Create temporary config for this algorithm
                temp_config = OptimizationConfig(algorithm=algorithm)
                temp_evaluator = StrategyEvaluator(temp_config)
                temp_orchestrator = OptimizationOrchestrator(temp_config)
                
                try:
                    result = temp_orchestrator.optimize_strategy(training_data, algorithm)
                    results[algorithm] = result
                except Exception as e:
                    self.logger.error(f"{algorithm} optimization failed: {str(e)}")
                    results[algorithm] = None
            else:
                self.logger.warning(f"Unknown algorithm: {algorithm}")
        
        return results
    
    def _save_optimization_result(self, result: OptimizationResult, algorithm: str):
        """Save optimization result to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_result_{algorithm}_{timestamp}.json"
        
        # Convert result to serializable format
        result_dict = {
            "best_strategy": result.best_strategy,
            "best_score": result.best_score,
            "optimization_history": result.optimization_history,
            "convergence_data": result.convergence_data,
            "performance_metrics": result.performance_metrics,
            "algorithm_used": result.algorithm_used,
            "optimization_time": getattr(result, 'optimization_time', None),
            "timestamp": result.timestamp
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Optimization result saved to {filename}")
    
    def create_optimization_report(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        """Create comprehensive optimization report"""
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "algorithms_tested": list(results.keys()),
            "comparison_results": {}
        }
        
        for algorithm, result in results.items():
            if result is not None:
                report["comparison_results"][algorithm] = {
                    "best_score": result.best_score,
                    "optimization_time": getattr(result, 'optimization_time', None),
                    "convergence_data": result.convergence_data,
                    "best_strategy": result.best_strategy,
                    "performance_metrics": result.performance_metrics
                }
            else:
                report["comparison_results"][algorithm] = {
                    "status": "failed",
                    "error": "Optimization failed to complete"
                }
        
        # Find best overall result
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_algorithm = max(valid_results.keys(), 
                               key=lambda a: valid_results[a].best_score)
            best_result = valid_results[best_algorithm]
            
            report["best_overall"] = {
                "algorithm": best_algorithm,
                "score": best_result.best_score,
                "strategy": best_result.best_strategy
            }
        
        return report
    
    def apply_optimized_strategy(self, 
                               strategy_config: Dict[str, Any],
                               data: List[Dict]) -> Dict[str, Any]:
        """Apply optimized strategy to data"""
        
        self.logger.info("Applying optimized augmentation strategy...")
        
        try:
            # Create augmentation config
            aug_config = self.evaluator._create_augmentation_config(strategy_config)
            
            # Create augmentor
            augmentor = DataAugmentor(aug_config)
            
            # Apply pipeline
            results = apply_augmentation_pipeline(data, aug_config, augmentor)
            
            # Evaluate quality
            quality_metrics = self.evaluator.quality_assessor.assess_data_quality(
                results.get("augmented_conversations", [])
            )
            
            results["optimization_quality_metrics"] = quality_metrics
            results["strategy_applied"] = strategy_config
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimized strategy: {str(e)}")
            raise


def optimize_augmentation_strategy(data_file: str,
                                 algorithm: str = "genetic",
                                 output_dir: str = "./optimization_results") -> Dict[str, Any]:
    """Convenience function to optimize augmentation strategy"""
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract training scenarios
    training_data = []
    if isinstance(data, dict):
        if "scenarios" in data:
            training_data = data["scenarios"]
        elif "conversations" in data:
            training_data = data["conversations"]
    
    if not training_data:
        raise ValueError("No training data found in input file")
    
    # Create optimization config
    config = OptimizationConfig(
        algorithm=algorithm,
        population_size=10,  # Reduced for demo
        generations=5,       # Reduced for demo
        evaluation_metric="overall_score"
    )
    
    # Run optimization
    orchestrator = OptimizationOrchestrator(config)
    result = orchestrator.optimize_strategy(training_data, algorithm)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save results
    orchestrator._save_optimization_result(result, algorithm)
    
    # Create report
    report = orchestrator.create_optimization_report({algorithm: result})
    
    report_file = Path(output_dir) / f"optimization_report_{algorithm}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return {
        "optimization_result": result,
        "optimization_report": report,
        "best_strategy": result.best_strategy,
        "best_score": result.best_score,
        "output_files": [str(report_file)]
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Augmentation Strategy Optimization Module")
    
    # Create sample data
    sample_data = [
        {
            "conversation": [
                {"speaker": "patient", "text": "I have chest pain"},
                {"speaker": "ai", "text": "Can you describe the pain?"}
            ]
        }
    ]
    
    # Run optimization
    config = OptimizationConfig(
        algorithm="random",
        population_size=5,
        generations=3
    )
    
    orchestrator = OptimizationOrchestrator(config)
    
    try:
        result = orchestrator.optimize_strategy(sample_data, "random")
        print(f"Optimization complete. Best score: {result.best_score:.4f}")
        print(f"Best strategy: {result.best_strategy}")
    except Exception as e:
        print(f"Optimization failed: {str(e)}")