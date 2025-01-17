class Config:
    """Configuration class for stock market data preprocessing."""

    def __init__(self):
        self.input_file = "data/simulated_stock_data_last_year.csv"
        self.output_file = "data/preprocessed_stock_data.csv"
        self.sequence_length = 10
        self.sma_window = 20
        self.rsi_window = 14
