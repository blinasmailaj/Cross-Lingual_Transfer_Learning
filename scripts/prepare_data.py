import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.data_loader import prepare_multilingual_data

def download_and_prepare_data():
    """
    Script to download and prepare dataset
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Initialize project configuration
        config = ProjectConfig()
        
        # Prepare data
        train_loader, val_loader, test_loader, tokenizer = prepare_multilingual_data(config)
        
        logger.info("Data preparation completed successfully")
        logger.info(f"Train loader size: {len(train_loader.dataset)}")
        logger.info(f"Validation loader size: {len(val_loader.dataset)}")
        logger.info(f"Test loader size: {len(test_loader.dataset)}")
    
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise

if __name__ == '__main__':
    download_and_prepare_data()
