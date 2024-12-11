<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

let socket = io();

// Initialize data arrays for the chart
let timestamps = [];
let prices = [];
let ma9 = [];
let ma21 = [];

// Create the initial chart
let trace1 = {
    x: timestamps,
    y: prices,
    name: 'Price',
    type: 'scatter'
};

let trace2 = {
    x: timestamps,
    y: ma9,
    name: '9-day MA',
    type: 'scatter'
};

let trace3 = {
    x: timestamps,
    y: ma21,
    name: '21-day MA',
    type: 'scatter'
};

let layout = {
    title: 'SPY Price and Moving Averages',
    xaxis: { title: 'Time' },
    yaxis: { title: 'Price' }
};

Plotly.newPlot('chart', [trace1, trace2, trace3], layout);

// Handle incoming data
socket.on('update_data', function(data) {
    // Update signal indicator
    const signalBox = document.getElementById('signal-indicator');
    if (data.entry === 2) {
        signalBox.className = 'signal-box buy-signal';
        signalBox.textContent = 'BUY SIGNAL!';
    } else if (data.entry === -2) {
        signalBox.className = 'signal-box sell-signal';
        signalBox.textContent = 'SELL SIGNAL!';
    } else {
        signalBox.className = 'signal-box no-signal';
        signalBox.textContent = 'No Signal';
    }

    // Update price display
    document.getElementById('latest-price').textContent = 
        `Latest Price: $${data.close.toFixed(2)} | Time: ${data.timestamp}`;

    // Update chart data
    timestamps.push(data.timestamp);
    prices.push(data.close);
    ma9.push(data['9_day_ma']);
    ma21.push(data['21_day_ma']);

    // Keep only the last 100 data points
    if (timestamps.length > 100) {
        timestamps.shift();
        prices.shift();
        ma9.shift();
        ma21.shift();
    }

    // Update the chart
    Plotly.update('chart', {
        x: [timestamps, timestamps, timestamps],
        y: [prices, ma9, ma21]
    });
}); 

// Add near the top of your file, where you initialize socket
const tickerSelect = document.getElementById('ticker-select');
let currentTicker = 'SPY';

// Add this event listener
tickerSelect.addEventListener('change', (event) => {
    currentTicker = event.target.value;
    // Emit event to server to change ticker
    socket.emit('change_ticker', { ticker: currentTicker });
    
    // Reset arrays and clear chart
    prices = [];
    ma9 = [];
    ma21 = [];
    Plotly.purge('chart');
}); 

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysis-form');
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results');
    const plotContainer = document.getElementById('plot-container');

    console.log('DOM loaded, form:', form); // Debug log

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        console.log('Form submitted'); // Debug log
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultsContainer.style.display = 'none';
        plotContainer.innerHTML = '';
        
        try {
            console.log('Sending fetch request...'); // Debug log
            const formData = new FormData(form);
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            console.log('Response received:', response); // Debug log
            const data = await response.json();
            console.log('Data:', data); // Debug log
            
            if (data.success) {
                console.log('Success, updating plots...'); // Debug log
                // Update plots
                plotContainer.innerHTML = data.plots.join('');
                
                // Update returns
                document.getElementById('actual-return').textContent = data.actual_return;
                document.getElementById('system-return').textContent = data.system_return;
                
                // Show results
                resultsContainer.style.display = 'block';
            } else {
                console.error('Error from server:', data.error); // Debug log
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the data.');
        } finally {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }
    });
}); 