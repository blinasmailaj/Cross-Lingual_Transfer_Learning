import os
import glob
import torch
import logging

logger = logging.getLogger(__name__)

def find_available_checkpoints(checkpoint_dir):
    logging.info(f'sddasd{checkpoint_dir}')
    """Find and validate available checkpoint files"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    logging.info(f'sddasd{checkpoint_files}')
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return []
        
    valid_checkpoints = []
    # for cf in checkpoint_files:
    #     try:
    #         # Try to load header only to validate
    #         torch.load(cf, map_location='cpu', weights_only=True)
    #         size_mb = os.path.getsize(cf)/1024/1024
    #         valid_checkpoints.append((cf, size_mb))
    #         logger.info(f"Found valid checkpoint: {cf} ({size_mb:.2f} MB)")
    #     except Exception as e:
    #         logger.warning(f"Invalid checkpoint found at {cf}: {str(e)}")
    # logging.info(f'sddasd{valid_checkpoints}')
    return checkpoint_files

def load_checkpoint(checkpoint_path, model, device):
    """Robust checkpoint loading with multiple fallback attempts"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        # First attempt: normal loading
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded successfully")
        return model, checkpoint
        
    except Exception as e:
        logger.warning(f"First loading attempt failed: {str(e)}")
        
        try:
            # Second attempt: CPU loading then move to device
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            logger.info("Checkpoint loaded successfully with CPU fallback")
            return model, checkpoint
            
        except Exception as e2:
            logger.warning(f"Second loading attempt failed: {str(e2)}")
            
            try:
                # Last resort: pickle loading
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location=device, pickle_module=pickle)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Checkpoint loaded successfully with pickle fallback")
                return model, checkpoint
                
            except Exception as e3:
                logger.error(f"All loading attempts failed: {str(e3)}")
                raise

def safe_save_checkpoint(model, optimizer, config, filepath):
    """Safely save checkpoint with temporary file"""
    temp_path = filepath + '.tmp'
    
    try:
        # Save to temporary file first
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, temp_path, _use_new_zipfile_serialization=True)
        
        # If save was successful, rename to final filename
        os.replace(temp_path, filepath)
        logger.info(f"Checkpoint saved successfully to {filepath}")
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Failed to save checkpoint: {str(e)}")
        raise