# BIST 100 Direction Prediction with Commodity Prices

A machine learning model that predicts the direction (increase/decrease) of the BIST 100 index using commodity prices, exchange rates, and financial indicators.

## About the Project

This project aims to predict whether the BIST 100 index will rise on the next trading day using machine learning techniques. The prediction is made using the XGBoost algorithm, and an interactive web interface has been developed with Streamlit.

### Data Sources

- BIST 100 index (XU100.IS)
- Gold Futures (GC=F)
- Crude Oil Futures (CL=F)
- USD/TRY Exchange Rate (USDTRY=X)
- US 10-Year Treasury Yield (^TNX)
- Natural Gas Futures (NG=F)
- VIX Volatility Index (^VIX)

All data is automatically retrieved from the Yahoo Finance API.

### Features

- Detailed visualization of global market data
- Correlation analyses (standard and rolling correlation)
- Lag analysis to detect delayed effects of different variables on BIST 100
- High-accuracy classification model with XGBoost
- Model performance metrics and evaluation tools
- Feature importance analysis
- Interactive prediction indicators and charts

## Installation and Running

### Requirements

- Python 3.7+
- pip or conda package manager

### Installation Steps

1. Clone the project:
```bash
git clone https://github.com/ayhannbozkurt/commodities-bist-analyze.git
cd commodities-bist-analyze
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Start data collection:
```bash
python data_collector.py
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
commodities-bist-analyze/
├── app.py                   # Streamlit web application main file
├── data.py                  # Data processing and preparation module
├── data_collector.py        # Data collection and download module
├── model.py                 # Model training, optimization, and evaluation module
├── model_trainer.py         # Model training process and metadata management
├── visualization.py         # Data visualization functions
├── requirements.txt         # Project dependencies
├── models/                  # Trained model and metadata files
└── data/                    # Data files directory
```

### Modules and Functions

#### `data_collector.py`
- Connects to Yahoo Finance API to automatically retrieve financial data
- Performs data cleaning and merging operations
- Saves raw data to the `data/` directory

#### `data.py`
- Performs data preprocessing and feature engineering
- Creates lag features
- Splits data into training and test sets
- Creates the target variable (next day's BIST 100 direction)

#### `model.py`
- Training functions for XGBoost model
- Model evaluation and performance measurement
- Hyperparameter optimization and selection of the best model
- Model saving and loading functions

#### `model_trainer.py`
- Automates the end-to-end model training process
- Model metadata management
- Evaluates success criteria

#### `visualization.py`
- Functions for visualizing data and model results
- Correlation analyses and heatmaps
- Interactive charts and dashboards
- Visualizations for lag analyses

#### `app.py`
- Streamlit web application
- User interface and page layout
- Model predictions and data exploration interface

## Usage

The application consists of 3 main tabs:

### 1. Market Data
- View market data over different time periods
- Correlation matrices and daily change charts
- Advanced correlation analyses

### 2. BIST 100 Prediction
- Next-day BIST 100 direction prediction with the latest data
- Prediction probability and confidence level
- Display of features the model finds important
- Accuracy rates of recent predictions

### 3. Global Variables and Lag Analysis
- Analysis of the effects of different commodities and financial indicators on BIST 100
- Detection of delayed effects of variables through lag analysis
- Rolling correlation analysis
- Global variables dashboard

## Technical Details

### Data Processing
- Daily percentage changes are used as basic features
- Features are created for different lag days (1, 10, 30)
- NaN values are filled using forward and backward filling methods

### Model
- XGBoost classifier is used
- Target variable: Next day's direction of BIST 100 (1: increase, 0: decrease)
- Models are evaluated with 5-fold cross-validation
- Performance is measured with accuracy rate, F1-score, and ROC curve

## Contributing

1. Fork this project
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push your branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
