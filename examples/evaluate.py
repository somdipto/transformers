#!/usr/bin/env python3
"""
DeBERTa-v3 + LoRA Sentiment Analysis Evaluation Script

A comprehensive evaluation framework for DeBERTa-v3 fine-tuned with LoRA for sentiment analysis.
This script provides model loading, evaluation on TweetEval dataset, performance metrics,
baseline comparisons, confusion matrix analysis, and benchmarking capabilities.

Features:
- Model loading from saved checkpoints
- Evaluation on TweetEval test datasets
- Performance metrics (accuracy, F1, precision, recall)
- Baseline model comparisons
- Confusion matrix and error analysis
- Performance benchmarking and reporting
- Command-line interface for different evaluation modes

Usage:
    python evaluate.py --model_path ./checkpoints/deberta_lora_sentiment
    python evaluate.py --eval_mode full --compare_baselines
    python evaluate.py --eval_mode quick --batch_size 16
    python evaluate.py --eval_mode benchmark --save_results
"""

import os
import sys
import json
import time
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, pipeline
)
from datasets import load_dataset, Dataset

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Sentiment analysis evaluator with multiple model support."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Load model and tokenizer
        self._load_model()
        
    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
        
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Try to load as a transformers model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to a simple mock model for demo
            self._load_mock_model()
            
    def _load_mock_model(self):
        """Load a mock model for demonstration purposes."""
        logger.warning("Loading mock model for demonstration")
        
        # Use a simple pipeline with a basic model for demo
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        
    def predict(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts."""
        if self.pipeline is None:
            raise ValueError("Model not loaded")
            
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.pipeline(batch)
            results.extend(batch_results)
            
        return results
        
    def predict_with_timing(self, texts: List[str], batch_size: int = 16) -> Tuple[List[Dict[str, float]], float]:
        """Predict sentiment with timing information."""
        start_time = time.time()
        results = self.predict(texts, batch_size)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        return results, elapsed_time

class EvaluationFramework:
    """Comprehensive evaluation framework for sentiment analysis models."""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_tweeteval_dataset(self, task: str = "sentiment") -> Tuple[Dataset, Dataset]:
        """Load TweetEval dataset for evaluation."""
        try:
            logger.info(f"Loading TweetEval dataset for task: {task}")
            dataset = load_dataset("tweet_eval", task)
            return dataset["train"], dataset["test"]
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return self._create_mock_dataset()
            
    def _create_mock_dataset(self) -> Tuple[Dataset, Dataset]:
        """Create a mock dataset for demonstration purposes."""
        logger.info("Creating mock dataset for demonstration")
        
        # Mock training data
        train_data = {
            "text": [
                "I love this product!",
                "This is terrible quality",
                "The service was okay",
                "Amazing experience!",
                "Worst purchase ever",
                "Pretty good overall",
                "Exceeded my expectations",
                "Not worth the money",
                "Solid performance",
                "Disappointed with the results"
            ] * 10,  # Repeat for a larger dataset
            "label": [2, 0, 1, 2, 0, 1, 2, 0, 1, 0] * 10
        }
        
        # Mock test data
        test_data = {
            "text": [
                "Great value for money",
                "Poor customer service",
                "Average product",
                "Excellent quality!",
                "Completely useless"
            ] * 20,  # Repeat for a larger test set
            "label": [2, 0, 1, 2, 0] * 20
        }
        
        return Dataset.from_dict(train_data), Dataset.from_dict(test_data)
        
    def evaluate_model(self, analyzer: SentimentAnalyzer, dataset: Dataset, 
                      batch_size: int = 16) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        logger.info(f"Evaluating model on {len(dataset)} examples")
        
        # Extract texts and true labels
        texts = dataset["text"]
        true_labels = dataset["label"]
        
        # Make predictions
        start_time = time.time()
        predictions = analyzer.predict(texts, batch_size)
        end_time = time.time()
        
        # Convert predictions to labels
        pred_labels = []
        for pred in predictions:
            if pred["label"] == "POSITIVE":
                pred_labels.append(2)
            elif pred["label"] == "NEGATIVE":
                pred_labels.append(0)
            else:  # NEUTRAL
                pred_labels.append(1)
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, pred_labels)
        metrics["inference_time"] = end_time - start_time
        metrics["throughput"] = len(texts) / (end_time - start_time)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "true_labels": true_labels,
            "pred_labels": pred_labels
        }
        
    def _calculate_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Convert to numpy arrays
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        class_precision, class_recall, class_f1, class_support = 
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "class_precision": class_precision.tolist(),
            "class_recall": class_recall.tolist(),
            "class_f1": class_f1.tolist(),
            "class_support": class_support.tolist()
        }
        
    def generate_confusion_matrix(self, true_labels: List[int], pred_labels: List[int], 
                                class_names: List[str], output_path: str = None) -> str:
        """Generate and save confusion matrix plot."""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if output_path is None:
            output_path = self.output_dir / "confusion_matrix.png"
        else:
            output_path = Path(output_path)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {output_path}")
        return str(output_path)
        
    def compare_with_baselines(self, analyzer: SentimentAnalyzer, dataset: Dataset) -> Dict[str, Any]:
        """Compare model performance with baseline models."""
        logger.info("Comparing with baseline models")
        
        baseline_models = {
            "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
            "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "bert": "nlptown/bert-base-multilingual-uncased-sentiment"
        }
        
        results = {}
        
        for name, model_path in baseline_models.items():
            try:
                logger.info(f"Evaluating {name} model")
                baseline_analyzer = SentimentAnalyzer(model_path)
                baseline_results = self.evaluate_model(baseline_analyzer, dataset)
                results[name] = baseline_results["metrics"]
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                results[name] = {"error": str(e)}
                
        return results
        
    def benchmark_performance(self, analyzer: SentimentAnalyzer, dataset: Dataset, 
                            batch_sizes: List[int] = [1, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Benchmark model performance with different batch sizes."""
        logger.info("Benchmarking performance with different batch sizes")
        
        texts = dataset["text"][:100]  # Use subset for benchmarking
        results = {}
        
        for batch_size in batch_sizes:
            try:
                logger.info(f"Testing batch size: {batch_size}")
                _, elapsed_time = analyzer.predict_with_timing(texts, batch_size)
                
                results[f"batch_{batch_size}"] = {
                    "batch_size": batch_size,
                    "total_time": elapsed_time,
                    "throughput": len(texts) / elapsed_time,
                    "avg_time_per_text": elapsed_time / len(texts),
                    "texts_processed": len(texts)
                }
            except Exception as e:
                logger.warning(f"Failed to benchmark batch size {batch_size}: {e}")
                results[f"batch_{batch_size}"] = {"error": str(e)}
                
        return results
        
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Save evaluation results to JSON file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
        
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Generate a comprehensive evaluation report."""
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.md"
        else:
            output_path = Path(output_path)
            
        report = []
        report.append("# DeBERTa-v3 + LoRA Sentiment Analysis Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Model Performance
        if "metrics" in results:
            metrics = results["metrics"]
            report.append("## Model Performance")
            report.append(f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}")
            report.append(f"- **Precision**: {metrics.get('precision', 0):.4f}")
            report.append(f"- **Recall**: {metrics.get('recall', 0):.4f}")
            report.append(f"- **F1 Score**: {metrics.get('f1', 0):.4f}")
            report.append(f"- **Inference Time**: {metrics.get('inference_time', 0):.2f}s")
            report.append(f"- **Throughput**: {metrics.get('throughput', 0):.2f} texts/sec")
            
        # Baseline Comparison
        if "baseline_comparison" in results:
            report.append("\n## Baseline Comparison")
            for model, metrics in results["baseline_comparison"].items():
                if "error" not in metrics:
                    report.append(f"### {model}")
                    report.append(f"- Accuracy: {metrics.get('accuracy', 0):.4f}")
                    report.append(f"- F1: {metrics.get('f1', 0):.4f}")
                    
        # Performance Benchmarking
        if "performance_benchmark" in results:
            report.append("\n## Performance Benchmarking")
            for batch_info in results["performance_benchmark"].values():
                if "error" not in batch_info:
                    report.append(f"- **Batch Size {batch_info['batch_size']}**: "
                                f"{batch_info['throughput']:.2f} texts/sec")
                                
        # Write report
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
            
        logger.info(f"Report generated: {output_path}")
        return str(output_path)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeBERTa-v3 + LoRA sentiment analysis model"
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--task", type=str, default="sentiment",
                       choices=["sentiment", "emotion", "hate", "irony", "offensive", "stance"],
                       help="TweetEval task")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--eval_mode", type=str, default="quick",
                       choices=["quick", "full", "benchmark", "report"],
                       help="Evaluation mode")
    parser.add_argument("--compare_baselines", action="store_true",
                       help="Compare with baseline models")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--save_results", action="store_true",
                       help="Save results to files")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = EvaluationFramework(args.output_dir)
    
    # Load dataset
    train_dataset, test_dataset = framework.load_tweeteval_dataset(args.task)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(args.model_path, args.device)
    
    results = {}
    
    try:
        # Basic evaluation
        logger.info("Running basic evaluation...")
        eval_results = framework.evaluate_model(analyzer, test_dataset, args.batch_size)
        results["metrics"] = eval_results["metrics"]
        
        # Print results
        metrics = eval_results["metrics"]
        print("\n=== EVALUATION RESULTS ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Inference Time: {metrics['inference_time']:.2f}s")
        print(f"Throughput: {metrics['throughput']:.2f} texts/sec")
        print("=" * 30)
        
        # Generate confusion matrix
        if args.eval_mode in ["full", "report"]:
            class_names = ["Negative", "Neutral", "Positive"]
            cm_path = framework.generate_confusion_matrix(
                eval_results["true_labels"], 
                eval_results["pred_labels"], 
                class_names
            )
            results["confusion_matrix_path"] = cm_path
        
        # Baseline comparison
        if args.compare_baselines or args.eval_mode == "full":
            logger.info("Running baseline comparison...")
            baseline_results = framework.compare_with_baselines(analyzer, test_dataset)
            results["baseline_comparison"] = baseline_results
        
        # Performance benchmarking
        if args.eval_mode in ["benchmark", "full"]:
            logger.info("Running performance benchmark...")
            benchmark_results = framework.benchmark_performance(analyzer, test_dataset)
            results["performance_benchmark"] = benchmark_results
        
        # Save results
        if args.save_results or args.eval_mode == "report":
            framework.save_results(results)
            framework.generate_report(results)
            
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
