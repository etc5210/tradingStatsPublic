import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ml_logic import StockPredictor

def download_data(ticker):
    # prepare data from 85 days ago to today
    time = datetime.now()
    time = time.strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=85)).strftime('%Y-%m-%d')
    
    # download historical data & clean it up
    data = yf.download(ticker, start=start_date, interval='1h', prepost=False)
    day = np.arange(1, len(data) + 1)
    data['day'] = day
    data.drop(columns=['Adj Close', 'Volume'], inplace=True)
    data = data[['day', 'Open', 'High', 'Low', 'Close']]
    
    return data

def prep_data(data, sensitive=False):
    # Calculate daily percentage change
    data['daily_change'] = data['Close'].pct_change() * 100
    
    # Set moving average periods based on sensitive flag
    short_period = 5 if sensitive else 9
    long_period = 13 if sensitive else 21
    
    # Calculate moving averages
    data[f'{short_period}-day'] = data['Close'].rolling(short_period).mean().shift(1)
    data[f'{long_period}-day'] = data['Close'].rolling(long_period).mean().shift(1)
    
    # Calculate signals
    data['signal'] = np.where(data[f'{short_period}-day'] > data[f'{long_period}-day'], 1, 0)
    data['signal'] = np.where(data[f'{short_period}-day'] < data[f'{long_period}-day'], -1, data['signal'])
    data['signal'] = np.where(data['daily_change'] > 10, 1, data['signal'])
    
    data.dropna(subset=[f'{short_period}-day', f'{long_period}-day'], inplace=True)
    
    data['return'] = np.log(data['Close']).diff()
    data['system_return'] = data['signal'] * data['return']
    data['entry'] = data.signal.diff()
    
    return data

def analyze_returns(actual_return, system_return):
    # Convert percentage strings to floats
    actual = float(actual_return.strip('%')) / 100
    system = float(system_return.strip('%')) / 100
    
    # Calculate the difference
    difference = system - actual
    
    if difference > 0:
        return f"The trading system outperformed buy-and-hold by {difference:.2%}"
    elif difference < 0:
        return f"The trading system underperformed buy-and-hold by {abs(difference):.2%}"
    else:
        return "The trading system performed exactly the same as buy-and-hold"

def generate_recommendation(data, ml_prediction=None):
    """Generate trading recommendation based on technical and ML signals"""
    # Get the latest signal and daily change
    latest_signal = data['signal'].iloc[-1]
    latest_change = data['daily_change'].iloc[-1]
    latest_close = data['Close'].iloc[-1]
    
    # Base recommendation on technical signals
    technical_signal = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "NEUTRAL"
    
    # Incorporate ML prediction if available
    if ml_prediction:
        pred_change = ml_prediction['predicted_change']
        pred_price = ml_prediction['predicted_price']
        
        # Combine technical and ML signals
        if pred_change > 1.0 and technical_signal != "SELL":
            recommendation = f"STRONG BUY - Technical: {technical_signal}, ML Predicts: ${pred_price:.2f} (↑{pred_change:.1f}%)"
        elif pred_change < -1.0 and technical_signal != "BUY":
            recommendation = f"STRONG SELL - Technical: {technical_signal}, ML Predicts: ${pred_price:.2f} (↓{abs(pred_change):.1f}%)"
        else:
            recommendation = f"HOLD - Technical: {technical_signal}, ML Predicts: ${pred_price:.2f} ({pred_change:+.1f}%)"
    else:
        # Fall back to original technical recommendation
        if latest_signal == 1:
            if latest_change > 0:
                recommendation = f"BUY - Upward trend detected. Latest close: ${latest_close:.2f} (↑{latest_change:.1f}%)"
            else:
                recommendation = f"HOLD - Potential buying opportunity on dip. Latest close: ${latest_close:.2f} (↓{abs(latest_change):.1f}%)"
        elif latest_signal == -1:
            if latest_change < 0:
                recommendation = f"SELL - Downward trend detected. Latest close: ${latest_close:.2f} (↓{abs(latest_change):.1f}%)"
            else:
                recommendation = f"HOLD - Consider taking profits. Latest close: ${latest_close:.2f} (↑{latest_change:.1f}%)"
        else:
            recommendation = f"NEUTRAL - No clear trend. Latest close: ${latest_close:.2f} ({latest_change:+.1f}%)"
    
    return recommendation

def analyze_technical(data, sensitive=False):
    """Perform technical analysis on the stock data"""
    try:
        # Prepare data with technical indicators
        processed_data = prep_data(data, sensitive)
        
        # Calculate latest technical signals
        latest_data = processed_data.iloc[-1]
        
        return {
            'data': processed_data,
            'latest_signal': latest_data['signal'],
            'latest_close': latest_data['Close'],
            'latest_change': latest_data['daily_change']
        }
        
    except Exception as e:
        return {'error': str(e)}

def analyze_stock(ticker, sensitive=False):
    try:
        # Download historical data
        data = download_data(ticker)
        data.name = ticker  # Set the name of the DataFrame to the ticker
        
        # Perform technical analysis
        result = analyze_technical(data, sensitive)
        
        if 'error' in result:
            return {'error': result['error']}
        
        # Generate recommendation without ML prediction for now
        recommendation = generate_recommendation(result['data'])
        
        return {
            'data': result['data'],
            'recommendation': recommendation
        }
        
    except Exception as e:
        return {'error': str(e)}

def generate_plot_data(data, ticker, sensitive=False):
    short_ma = '5-day' if sensitive else '9-day'
    long_ma = '13-day' if sensitive else '21-day'
    
    plots = []
    
    # Price and signals plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=ticker))
    fig.add_trace(go.Scatter(x=data.index, y=data[short_ma], name=f'{short_ma[0:-4]}-day'))
    fig.add_trace(go.Scatter(x=data.index, y=data[long_ma], name=f'{long_ma[0:-4]}-day'))
    
    # Add buy/sell markers
    fig.add_trace(go.Scatter(
        x=data.loc[data.entry == 2].index,
        y=data[short_ma][data.entry == 2],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12, color='green'),
        name='Buy Signal'
    ))
    fig.add_trace(go.Scatter(
        x=data.loc[data.entry == -2].index,
        y=data[long_ma][data.entry == -2],
        mode='markers',
        marker=dict(symbol='triangle-down', size=12, color='red'),
        name='Sell Signal'
    ))
    
    fig.update_layout(
        title={
            'text': f'Historical Signals for {ticker} last 60 days',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0, y=1),
        margin=dict(t=100, l=50, r=50, b=50),  # Add margins
        paper_bgcolor='white',  # Add white background
        plot_bgcolor='white',   # Add white background
        width=1000,            # Set width
        height=600             # Set height
    )
    
    plots.append(fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',  # Use CDN version of plotly
        config={'displayModeBar': True}  # Show the modebar
    ))
    
    # Returns plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=data.index,
        y=np.exp(data['return']).cumprod(),
        name='Buy/Hold'
    ))
    fig2.add_trace(go.Scatter(
        x=data.index,
        y=np.exp(data['system_return']).cumprod(),
        name='System'
    ))
    
    fig2.add_hline(y=1, line_dash="dash", line_color="black", line_width=1, opacity=0.3)
    
    fig2.update_layout(
        title={
            'text': f'Returns Comparison for {ticker} last 60 days',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title='Date',
        yaxis_title='Returns',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=0, y=1),
        margin=dict(t=100, l=50, r=50, b=50),  # Add margins
        paper_bgcolor='white',  # Add white background
        plot_bgcolor='white'
    )
    
    plots.append(fig2.to_html(full_html=False))
    
    # Calculate returns
    total_return = np.exp(data['return']).cumprod()[-1]
    system_return = np.exp(data['system_return']).cumprod()[-1]
    
    # Generate recommendation
    recommendation = generate_recommendation(data)
    
    return {
        'plots': plots,
        'actual_return': f'{total_return:.2%}',
        'system_return': f'{system_return:.2%}',
        'recommendation': recommendation  # Add recommendation to return dict
    }

    # Original matplotlib code (commented out)
    """
    plots = []
    
    # Price and signals plot
    plt.figure(figsize=(12, 6))
    plt.grid(True, alpha=.3)
    plt.plot(data['Close'], label=ticker)
    plt.plot(data[short_ma], label=f'{short_ma[0:-4]}-day')
    plt.plot(data[long_ma], label=f'{long_ma[0:-4]}-day')
    plt.plot(data.loc[data.entry == 2].index, data[short_ma][data.entry == 2], '^',
            color='g', markersize=12)
    plt.plot(data.loc[data.entry == -2].index, data[long_ma][data.entry == -2], 'v',
            color='r', markersize=12)
    plt.legend(loc=2)
    plt.title(f'Historical Signals for {ticker} last 60 days')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    
    # Returns plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.exp(data['return']).cumprod(), label='Buy/Hold')
    plt.plot(np.exp(data['system_return']).cumprod(), label='System')
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    plt.legend(loc=2)
    plt.grid(True, alpha=.6)
    plt.title(f'Returns for {ticker} last 60 days')
    
    # Save to base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plots.append(base64.b64encode(img.getvalue()).decode())
    plt.close()
    """