# Stock Price Forecasting with PMDARIMA

This Streamlit web application predicts stock prices using the PMDARIMA library, leveraging AutoARIMA to forecast future trends based on historical stock data.

## Description

This app fetches historical stock data from Yahoo Finance based on user input (stock symbol and period). It performs pre-modeling analysis, determines the differencing term, fits an AutoARIMA model, and displays the actual vs. predicted stock prices along with confidence intervals.

## Installation

1. Ensure you have Python installed.
2. Clone this repository:

    ```bash
    git clone https://github.com/your-username/stock-price-forecasting.git
    ```

3. Navigate to the project directory and install the required dependencies:

    ```bash
    cd stock-price-forecasting
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the application through your web browser at [http://localhost:8501](http://localhost:8501).

## How to Use

- **Enter Stock Symbol and Period**: Input the stock symbol (e.g., AAPL, MSFT) and the desired period (e.g., 10y for 10 years' data).
- **Data Display**: Shows the retrieved historical stock data in a tabular format.
- **Autocorrelation Plot**: Visualizes the autocorrelation of the stock prices at different lag times.
- **Estimating Differencing Term**: Estimates the differencing term required for modeling.
- **Fitting and Updating the Model**: Utilizes AutoARIMA to fit the model and update it iteratively with new observations.
- **Performance Metrics**: Calculates Mean Squared Error (MSE) and Symmetric Mean Absolute Percentage Error (SMAPE) to evaluate model performance.
- **Predictions and Confidence Intervals**: Displays predicted stock prices with confidence intervals compared to actual prices.

## Technologies Used

- Streamlit
- Pandas
- NumPy
- Matplotlib
- PMDARIMA
- yfinance
- scikit-learn

## Credits

This application utilizes the following libraries and tools:
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PMDARIMA](https://github.com/pmdarima/pmdarima)
- [yfinance](https://pypi.org/project/yfinance/)
- [scikit-learn](https://scikit-learn.org/)

## Webapp

[stockpredictionarima](https://stockpredictionarima.streamlit.app/)
