from flask import Flask, render_template, jsonify, request
from trading_logic import download_data, prep_data, generate_plot_data, analyze_returns, generate_recommendation, analyze_stock
from ml_logic import StockPredictor
import asyncio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', 'AAPL')
    sensitive = request.form.get('sensitive', 'false').lower() == 'true'
    
    try:
        result = analyze_stock(ticker, sensitive)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
            
        plot_data = generate_plot_data(result['data'], ticker, sensitive)
        
        return jsonify({
            'success': True,
            'plots': plot_data['plots'],
            'actual_return': plot_data['actual_return'],
            'system_return': plot_data['system_return'],
            'recommendation': result['recommendation']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker', 'AAPL')
    
    try:
        predictor = StockPredictor()
        data = predictor.train(ticker)
        prediction = predictor.predict_next_day(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)