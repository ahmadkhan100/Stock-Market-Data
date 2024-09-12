import logging
from stock_data_preprocessor import StockPreprocessor
from config import Config

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='stock_preprocessing.log',
        filemode='w'  # Overwrite the log file each time
    )

def main():
    """
    Main function to run the stock market data preprocessing pipeline.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting stock market data preprocessing pipeline")

    try:
        config = Config()
        preprocessor = StockPreprocessor(config)
        X, y, preprocessed_df = preprocessor.preprocess()

        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info("Preprocessing completed successfully.")

        # Save preprocessed data
        preprocessor.save_preprocessed_data(preprocessed_df)
        logger.info(f"Preprocessed data saved to {config.output_file}")

    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    main()
