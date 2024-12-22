# train.py
import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List
import numpy as np
from utils.metrics import compute_rouge_scores
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        
        # Initialize tracking metrics
        self.best_val_loss = float('inf')
        self.best_rouge_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.rouge_scores = []
        self.current_patience = 0
        
        # Setup training parameters
        self.num_training_steps = len(train_loader) * config['num_epochs']
        self.num_warmup_steps = int(self.num_training_steps * config['warmup_ratio'])
        
        # Initialize scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        # Create output directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Setup curriculum learning
        self.curriculum = {
            'epoch': 0,
            'max_length': config['max_source_length'] // 2,
            'difficulty': 0.0
        }
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'rouge_scores': [],
            'learning_rates': []
        }

    def save_checkpoint(self, epoch: int, val_loss: float, rouge_scores: Dict[str, float], checkpoint_type: str = 'best') -> str:
        """Save model checkpoint with metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f'{checkpoint_type}_model_{timestamp}.pt'
        )
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'rouge_scores': rouge_scores,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'curriculum': self.curriculum
        }, checkpoint_path)
        
        logger.info(f"Saved {checkpoint_type} checkpoint to {checkpoint_path}")
        return checkpoint_path

    def _update_curriculum(self, epoch: int) -> None:
        """Update curriculum learning parameters"""
        if epoch > self.curriculum['epoch']:
            self.curriculum['epoch'] = epoch
            # Gradually increase difficulty
            self.curriculum['difficulty'] = min(1.0, epoch / (self.config['num_epochs'] * 0.7))
            # Gradually increase max sequence length
            self.curriculum['max_length'] = int(
                self.config['max_source_length'] * 
                (0.5 + 0.5 * self.curriculum['difficulty'])
            )
            logger.info(f"Updated curriculum - Difficulty: {self.curriculum['difficulty']:.2f}, "
                       f"Max Length: {self.curriculum['max_length']}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_steps = 0
        
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}") as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    
                    # Get loss from outputs
                    loss = outputs.loss / self.config['gradient_accumulation_steps']
                    
                    # Backward pass
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['max_grad_norm']
                        )
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += loss.item()
                    epoch_steps += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    # Memory cleanup
                    del outputs, loss
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM in batch {batch_idx}, skipping...")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        continue
                    raise e
        
        return total_loss / epoch_steps
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation with metrics"""
        self.model.eval()
        total_val_loss = 0
        val_steps = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc="Validation") as pbar:
                for batch in self.val_loader:
                    try:
                        # Move batch to device
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        
                        # Forward pass with mixed precision
                        with autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss
                        
                        # Generate summaries
                        generated_ids = self.model.generate_summary(
                            batch['input_ids'],
                            batch['attention_mask'],
                            language_id=batch['language_id']
                        )
                        
                        # Decode predictions and references
                        predictions = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                        references = self.tokenizer.batch_decode(
                            batch['labels'],
                            skip_special_tokens=True
                        )
                        
                        all_predictions.extend(predictions)
                        all_references.extend(references)
                        
                        total_val_loss += loss.item()
                        val_steps += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("OOM during validation, skipping batch")
                            torch.cuda.empty_cache()
                            continue
                        raise e
        
        # Calculate metrics
        avg_val_loss = total_val_loss / val_steps
        rouge_scores = compute_rouge_scores(all_predictions, all_references)
        
        return avg_val_loss, rouge_scores

    def train(self) -> None:
        """Main training loop"""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Total training steps: {self.num_training_steps}")
        
        try:
            for epoch in range(self.config['num_epochs']):
                # Training phase
                avg_train_loss = self.train_epoch(epoch)
                self.train_losses.append(avg_train_loss)
                
                # Validation phase
                avg_val_loss, rouge_scores = self.validate()
                self.val_losses.append(avg_val_loss)
                self.rouge_scores.append(rouge_scores)
                
                # Update metrics history
                self.metrics_history['train_loss'].append(avg_train_loss)
                self.metrics_history['val_loss'].append(avg_val_loss)
                self.metrics_history['rouge_scores'].append(rouge_scores)
                self.metrics_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
                
                # Log metrics
                logger.info(f"\nEpoch {epoch + 1} metrics:")
                logger.info(f"Average training loss: {avg_train_loss:.4f}")
                logger.info(f"Average validation loss: {avg_val_loss:.4f}")
                logger.info("ROUGE Scores:")
                for metric, score in rouge_scores.items():
                    logger.info(f"{metric}: {score:.4f}")
                
                # Save visualization
                self.visualizer.plot_training_progress(
                    self.train_losses,
                    self.val_losses,
                    list(range(1, epoch + 2))
                )
                self.visualizer.plot_rouge_scores(rouge_scores)
                
                # Check for best model
                rouge_avg = np.mean(list(rouge_scores.values()))
                if rouge_avg > self.best_rouge_score:
                    self.best_rouge_score = rouge_avg
                    self.current_patience = 0
                    self.save_checkpoint(epoch, avg_val_loss, rouge_scores, 'best')
                else:
                    self.current_patience += 1
                
                # Early stopping check
                if self.current_patience >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
            # Save final model
            self.save_checkpoint(
                self.config['num_epochs'],
                avg_val_loss,
                rouge_scores,
                'final'
            )
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(
                epoch,
                avg_val_loss,
                rouge_scores,
                'interrupted'
            )
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
        finally:
            # Save training metrics
            metrics_path = os.path.join(self.config['output_dir'], 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        logger.info("Training completed!")

def train_model(
    config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: Any,
) -> None:
    """Wrapper function to initialize and run training"""
    trainer = TrainingManager(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        tokenizer,
    )
    trainer.train()