#!/usr/bin/env python3
"""
DeBERTa-v3 + LoRA Sentiment Analysis Training Script

This script provides a complete training pipeline for fine-tuning DeBERTa-v3 models
with LoRA (Low-Rank Adaptation) on sentiment analysis tasks using the TweetEval dataset.

Features:
- LoRA parameter-efficient fine-tuning
- EDA (Easy Data Augmentation) for improved performance
- Multiple training configurations and presets
- Comprehensive validation and evaluation
- Model checkpointing and saving
- Real-time training monitoring

Usage:
    python train.py --task sentiment --epochs 3 --output_dir ./results
    python train.py --task emotion --config high_accuracy --output_dir ./results
    python train.py --gpu_memory_gb 8 --augment --n_aug 4
"""

import os
import sys
import json
import torch
import argparse
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from transformers import set_seed
    from datasets import load_dataset, Dataset
    import peft
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Please install: pip install transformers datasets peft torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training with optimized defaults."""
    
    # Model Configuration
    model_name_or_path: str = field(default="microsoft/deberta-v3-base")
    num_labels: int = field(default=3)
    max_length: int = field(default=512)
    
    # LoRA Configuration
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "attention.self.query",
        "attention.self.key", 
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense"
    ])
    
    # Training Configuration
    task: str = field(default="sentiment")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_steps: int = field(default=500)
    max_steps: int = field(default=-1)
    
    # Data Augmentation
    augment: bool = field(default=False)
    n_aug: int = field(default=4)
    alpha_sr: float = field(default=0.1)
    alpha_ri: float = field(default=0.1)
    alpha_rs: float = field(default=0.1)
    p_rd: float = field(default=0.1)
    
    # Output Configuration
    output_dir: str = field(default="./results")
    logging_dir: str = field(default="./logs")
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_f1")
    
    # System Configuration
    seed: int = field(default=42)
    fp16: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    
    # Advanced Configuration
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.lora_rank > 0, "LoRA rank must be positive"
        assert self.lora_alpha > 0, "LoRA alpha must be positive"
        assert 0 <= self.lora_dropout <= 1, "LoRA dropout must be between 0 and 1"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_length > 0, "Max length must be positive"
        
        if self.augment:
            assert self.n_aug > 0, "Number of augmentations must be positive"
            assert 0 < self.alpha_sr <= 1, "Alpha for synonym replacement must be between 0 and 1"
            assert 0 < self.alpha_ri <= 1, "Alpha for random insertion must be between 0 and 1"
            assert 0 < self.alpha_rs <= 1, "Alpha for random swap must be between 0 and 1"
            assert 0 < self.p_rd <= 1, "Probability for random deletion must be between 0 and 1"    

def setup_config(preset: Optional[str] = None, **kwargs) -> TrainingConfig:
    """Setup training configuration with optional presets."""
    
    presets = {
        "default": TrainingConfig(),
        "high_accuracy": TrainingConfig(
            lora_rank=32,
            num_train_epochs=5,
            learning_rate=1e-4,
            augment=True,
            n_aug=8
        ),
        "fast_training": TrainingConfig(
            lora_rank=8,
            num_train_epochs=2,
            learning_rate=5e-4,
            per_device_train_batch_size=16,
            augment=False
        ),
        "memory_efficient": TrainingConfig(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            fp16=True,
            gradient_checkpointing=True
        )
    }
    
    if preset and preset in presets:
        config = presets[preset]
    else:
        config = TrainingConfig()
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config

def load_tweeteval_dataset(task: str, tokenizer, max_length: int, augment: bool = False, 
                         n_aug: int = 4, alpha: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Load and preprocess TweetEval dataset."""
    
    logger.info(f"Loading TweetEval dataset for task: {task}")
    
    # Load dataset
    dataset = load_dataset("tweet_eval", task)
    
    # Task-specific label mappings
    task_mappings = {
        "sentiment": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2},
        "emotion": {"anger": 0, "joy": 1, "sadness": 2},
        "hate": {"non-hate": 0, "hate": 1},
        "irony": {"non-irony": 0, "irony": 1},
        "offensive": {"non-offensive": 0, "offensive": 1},
        "stance": {"none": 0, "favor": 1, "against": 2}
    }
    
    if task not in task_mappings:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: {list(task_mappings.keys())}")
    
    label_map = task_mappings[task]
    
    def preprocess_function(examples):
        """Preprocess examples for training."""
        texts = examples["text"]
        
        # Apply EDA augmentation if enabled
        if augment and n_aug > 0:
            texts = apply_eda_augmentation(texts, n_aug, alpha)
        
        # Tokenize texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Map labels
        labels = [label_map.get(label, 0) for label in examples["label"]]
        tokenized["labels"] = labels
        
        return tokenized
    
    # Apply preprocessing
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    test_dataset = dataset["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    
    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_dataset, test_dataset

def apply_eda_augmentation(texts: List[str], n_aug: int = 4, alpha: float = 0.1) -> List[str]:
    """Apply EDA (Easy Data Augmentation) techniques."""
    
    # Simple EDA implementation for demonstration
    # In practice, you would use libraries like nlpaug or textattack
    
    augmented_texts = []
    
    for text in texts:
        # Add original text
        augmented_texts.append(text)
        
        # Add simple variations (word-level operations)
        words = text.split()
        
        for _ in range(n_aug - 1):  # -1 because we already added original
            if len(words) < 2:
                augmented_texts.append(text)
                continue
                
            # Simple synonym replacement (replace random word with common alternatives)
            import random
            word_idx = random.randint(0, len(words) - 1)
            
            # Simple replacements (in practice, use WordNet or pre-trained embeddings)
            synonym_map = {
                "good": "great", "bad": "terrible", "nice": "wonderful",
                "amazing": "fantastic", "awful": "horrible", "love": "adore",
                "hate": "detest", "like": "enjoy", "wonderful": "marvelous"
            }
            
            word = words[word_idx].lower()
            if word in synonym_map:
                new_word = synonym_map[word]
                if word.isupper():
                    new_word = new_word.upper()
                elif word[0].isupper():
                    new_word = new_word.capitalize()
                
                new_words = words.copy()
                new_words[word_idx] = new_word
                augmented_texts.append(" ".join(new_words))
            else:
                augmented_texts.append(text)
    
    return augmented_texts

def create_lora_model(model_name_or_path: str, config: TrainingConfig):
    """Create DeBERTa-v3 model with LoRA configuration."""
    
    logger.info("Creating DeBERTa-v3 model with LoRA configuration")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=config.num_labels,
        problem_type="single_label_classification"
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def create_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """Create training arguments for the Trainer."""
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    return TrainingArguments(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        report_to=["tensorboard"],
        logging_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps"
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description="Train DeBERTa-v3 + LoRA sentiment analysis model"
    )
    
    # Add all configuration parameters as command line arguments
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--task", type=str, default="sentiment", 
                       choices=["sentiment", "emotion", "hate", "irony", "offensive", "stance"])
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    
    # Data augmentation
    parser.add_argument("--augment", action="store_true", help="Enable EDA augmentation")
    parser.add_argument("--n_aug", type=int, default=4, help="Number of augmentations per example")
    parser.add_argument("--alpha_sr", type=float, default=0.1, help="Alpha for synonym replacement")
    parser.add_argument("--alpha_ri", type=float, default=0.1, help="Alpha for random insertion")
    parser.add_argument("--alpha_rs", type=float, default=0.1, help="Alpha for random swap")
    parser.add_argument("--p_rd", type=float, default=0.1, help="Probability for random deletion")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    # Advanced parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                       help="Enable gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Configuration presets
    parser.add_argument("--config", type=str, default=None,
                       choices=["default", "high_accuracy", "fast_training", "memory_efficient"],
                       help="Configuration preset")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup configuration
    config = setup_config(
        preset=args.config,
        model_name_or_path=args.model_name_or_path,
        task=args.task,
        num_labels=args.num_labels,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        augment=args.augment,
        n_aug=args.n_aug,
        alpha_sr=args.alpha_sr,
        alpha_ri=args.alpha_ri,
        alpha_rs=args.alpha_rs,
        p_rd=args.p_rd,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm
    )
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info(f"  Task: {config.task}")
    logger.info(f"  Model: {config.model_name_or_path}")
    logger.info(f"  LoRA Rank: {config.lora_rank}")
    logger.info(f"  LoRA Alpha: {config.lora_alpha}")
    logger.info(f"  LoRA Dropout: {config.lora_dropout}")
    logger.info(f"  Augmentation: {config.augment}")
    logger.info(f"  Output Directory: {config.output_dir}")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info("Loading and preprocessing dataset...")
    train_dataset, test_dataset = load_tweeteval_dataset(
        config.task, tokenizer, config.max_length, 
        config.augment, config.n_aug, config.alpha_sr
    )
    
    # Create model
    logger.info("Creating model with LoRA configuration...")
    model = create_lora_model(config.model_name_or_path, config)
    
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        # Save configuration
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(config.output_dir, "training_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Model saved to: {config.output_dir}")
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Final Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save evaluation results
        with open(os.path.join(config.output_dir, "eval_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Training script completed successfully!")

if __name__ == "__main__":
    main()
