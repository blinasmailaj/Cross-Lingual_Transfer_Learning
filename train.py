import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from typing import Dict, Any
from utils.metrics import compute_rouge_scores
import numpy as np

def train(
    config: Dict[str, Any],
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: Any,
    visualizer: Any
) -> None:
    """
    Main training loop.
    Args:
        config: Configuration dictionary
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        device: Device to train on (cpu/cuda)
        tokenizer: Tokenizer for text processing
        visualizer: Visualization utility object
    """
    # Setup training parameters
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    # Initialize scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs = []
    patience_counter = 0
    global_step = 0
    
    print(f"Starting training on device: {device}")
    print(f"Total training steps: {num_training_steps}")
    
    try:
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            
            # Training phase
            model.train()
            total_loss = 0
            epoch_steps = 0
            
            with tqdm(total=len(train_loader), desc="Training") as pbar:
                for i, batch in enumerate(train_loader):
                    try:
                        # Move batch to device
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = model(**batch)
                        loss = outputs.loss / config['gradient_accumulation_steps']
                        
                        # Backward pass
                        loss.backward()
                        
                        # Update weights if gradient accumulation steps reached
                        if (i + 1) % config['gradient_accumulation_steps'] == 0:
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 
                                config['max_grad_norm']
                            )
                            
                            # Optimizer step
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            
                            global_step += 1
                        
                        # Update progress
                        total_loss += loss.item()
                        epoch_steps += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'lr': scheduler.get_last_lr()[0]
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
                            print("\nWARNING: out of memory, skipping batch")
                            optimizer.zero_grad()
                            continue
                        else:
                            raise e
            
            # Calculate average training loss
            avg_train_loss = total_loss / epoch_steps
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_steps = 0
            all_predictions = []
            all_references = []
            
            print("\nRunning validation...")
            with torch.no_grad():
                with tqdm(total=len(val_loader), desc="Validation") as pbar:
                    for batch in val_loader:
                        try:
                            # Move batch to device
                            batch = {k: v.to(device) for k, v in batch.items()}
                            
                            # Forward pass
                            outputs = model(**batch)
                            loss = outputs.loss
                            
                            # Generate predictions
                            generated_ids = model.generate_summary(
                                batch['input_ids'],
                                batch['attention_mask']
                            )
                            
                            # Decode predictions and references
                            predictions = tokenizer.batch_decode(
                                generated_ids, 
                                skip_special_tokens=True
                            )
                            references = tokenizer.batch_decode(
                                batch['labels'],
                                skip_special_tokens=True
                            )
                            
                            # Collect predictions and references
                            all_predictions.extend(predictions)
                            all_references.extend(references)
                            
                            # Update loss
                            total_val_loss += loss.item()
                            val_steps += 1
                            
                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({'loss': loss.item()})
                            
                            # Memory cleanup
                            del outputs
                            del loss
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                                
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                                print("\nWARNING: out of memory during validation, skipping batch")
                                continue
                            else:
                                raise e
            
            # Calculate average validation loss and ROUGE scores
            avg_val_loss = total_val_loss / val_steps
            val_losses.append(avg_val_loss)
            epochs.append(epoch + 1)
            
            rouge_scores = compute_rouge_scores(all_predictions, all_references)
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} metrics:")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            print("ROUGE Scores:", rouge_scores)
            
            # Visualize progress
            visualizer.plot_training_progress(train_losses, val_losses, epochs)
            visualizer.plot_rouge_scores(rouge_scores)
            
            # Save checkpoint if best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': config
                }, checkpoint_path)
                
                print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Memory cleanup between epochs
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Final memory cleanup
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    print("\nTraining completed!")
    return train_losses, val_losses, rouge_scores

def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        tokenizer: Tokenizer for text processing
        device: Device to evaluate on
        config: Configuration dictionary
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    print("\nStarting evaluation...")
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
            for batch in test_loader:
                try:
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Generate predictions
                    generated_ids = model.generate_summary(
                        batch['input_ids'],
                        batch['attention_mask']
                    )
                    
                    # Decode predictions and references
                    predictions = tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    references = tokenizer.batch_decode(
                        batch['labels'],
                        skip_special_tokens=True
                    )
                    
                    # Collect predictions and references
                    all_predictions.extend(predictions)
                    all_references.extend(references)
                    
                    # Update loss
                    total_loss += loss.item()
                    
                    # Update progress bar
                    pbar.update(1)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        print("\nWARNING: out of memory during evaluation, skipping batch")
                        continue
                    else:
                        raise e
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    rouge_scores = compute_rouge_scores(all_predictions, all_references)
    
    # Combine metrics
    metrics = {
        'test_loss': avg_loss,
        **rouge_scores
    }
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics