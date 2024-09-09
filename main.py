import logging
from stock_preprocessor import StockPreprocessor
from config import Config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='stock_preprocessing.log'
    )

def main():
    """Main function to run the stock market data preprocessing pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting stock market data preprocessing pipeline")

    try:
        config = Config()
        preprocessor = StockPreprocessor(config)
        X, y, preprocessed_df = preprocessor.preprocess()

        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Preprocessed dataframe head:\n{preprocessed_df.head()}")

        # Save preprocessed data
        preprocessed_df.to_csv(config.output_file)
        logger.info(f"Preprocessed data saved to {config.output_file}")

    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()

