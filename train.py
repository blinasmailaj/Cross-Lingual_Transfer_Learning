import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from typing import Dict, Any, Tuple, List
from utils.metrics import compute_rouge_scores
import numpy as np
import logging
import json
from datetime import datetime

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
        visualizer: Any
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.tokenizer = tokenizer
        self.visualizer = visualizer
        
        # Initialize tracking metrics
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.all_metrics = []
        
        # Setup training parameters
        self.num_training_steps = len(train_loader) * config['num_epochs']
        self.num_warmup_steps = int(self.num_training_steps * config['warmup_ratio'])
        
        # Initialize scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        
        # Create directories if they don't exist
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['output_dir'], exist_ok=True)

    def save_checkpoint(self, epoch: int, val_loss: float, checkpoint_type: str = 'best') -> str:
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'], 
            f'{checkpoint_type}_model_{timestamp}.pt'
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'val_loss': val_loss,
            'training_metrics': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'epochs': self.epochs,
                'all_metrics': self.all_metrics
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {checkpoint_type} model checkpoint to {checkpoint_path}")
        
        # Save training metrics separately for easy access
        metrics_path = os.path.join(
            self.config['output_dir'], 
            f'training_metrics_{timestamp}.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(checkpoint['training_metrics'], f, indent=4)
        
        return checkpoint_path

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_steps = 0
        
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}") as pbar:
            for i, batch in enumerate(self.train_loader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config['gradient_accumulation_steps']
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights if gradient accumulation steps reached
                    if (i + 1) % self.config['gradient_accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['max_grad_norm']
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    # Update metrics
                    total_loss += loss.item()
                    epoch_steps += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'lr': self.scheduler.get_last_lr()[0]
                    })
                    
                    # Memory cleanup
                    del outputs
                    del loss
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        logger.warning("Out of memory, skipping batch")
                        self.optimizer.zero_grad()
                        continue
                    raise e
        
        return total_loss / epoch_steps

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation"""
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
                        
                        # Forward pass
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        
                        # Generate predictions
                        generated_ids = self.model.generate_summary(
                            batch['input_ids'],
                            batch['attention_mask']
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
                        pbar.set_postfix({'loss': loss.item()})
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            logger.warning("Out of memory during validation, skipping batch")
                            continue
                        raise e
        
        avg_val_loss = total_val_loss / val_steps
        rouge_scores = compute_rouge_scores(all_predictions, all_references)
        
        return avg_val_loss, rouge_scores

    def train(self) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        """Main training loop"""
        logger.info(f"Starting training on device: {self.device}")
        logger.info(f"Total training steps: {self.num_training_steps}")
        
        patience_counter = 0
        best_checkpoint_path = None
        
        try:
            for epoch in range(self.config['num_epochs']):
                # Training phase
                avg_train_loss = self.train_epoch(epoch)
                self.train_losses.append(avg_train_loss)
                
                # Validation phase
                avg_val_loss, rouge_scores = self.validate()
                self.val_losses.append(avg_val_loss)
                self.epochs.append(epoch + 1)
                self.all_metrics.append(rouge_scores)
                
                # Print metrics
                logger.info(f"\nEpoch {epoch + 1} metrics:")
                logger.info(f"Average training loss: {avg_train_loss:.4f}")
                logger.info(f"Average validation loss: {avg_val_loss:.4f}")
                logger.info("ROUGE Scores:", rouge_scores)
                
                # Visualize progress
                self.visualizer.plot_training_progress(
                    self.train_losses,
                    self.val_losses,
                    self.epochs
                )
                self.visualizer.plot_rouge_scores(rouge_scores)
                
                # Save checkpoint if best model
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_checkpoint_path = self.save_checkpoint(
                        epoch,
                        avg_val_loss,
                        'best'
                    )
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
            # Save final model
            final_checkpoint_path = self.save_checkpoint(
                self.config['num_epochs'], 
                avg_val_loss,
                'final'
            )
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            interrupt_checkpoint_path = self.save_checkpoint(
                epoch,
                avg_val_loss,
                'interrupted'
            )
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
        finally:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        logger.info("Training completed!")
        return self.train_losses, self.val_losses, self.all_metrics

def train_model(
    config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: Any,
    visualizer: Any
) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    """Wrapper function to initialize and run training"""
    trainer = TrainingManager(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        tokenizer,
        visualizer
    )
    return trainer.train()