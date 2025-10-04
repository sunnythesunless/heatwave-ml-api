from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import requests
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ML model
model = joblib.load('notebook/heatwave_prediction_model.pkl')

# Direct API key for production
API_KEY = "30558a9f8d57206a332a1321d9972735"


# Add error check
if not API_KEY:
    print("ERROR: WEATHER_API_KEY not found in environment variables")

def get_heatwave_percentage(weather_data):
    """Convert weather data to percentage prediction with correct feature order"""
    
    # EXACT SAME ORDER as training - this is critical!
    feature_order = [
        'temp_c', 'feelslike_c', 'humidity', 'dewpoint_c', 
        'pressure_mb', 'wind_kph', 'hour', 'month',
        'temp_rolling_3h', 'temp_rolling_24h', 'daily_temp_range'
    ]
    
    # Create DataFrame with exact feature order
    weather_df = pd.DataFrame([weather_data])[feature_order]
    
    probabilities = model.predict_proba(weather_df)
    return probabilities[0][1] * 100

def process_daily_weather(forecast_data):
    daily = {}
    for item in forecast_data['list']:
        date = datetime.fromtimestamp(item['dt']).date()
        ds = date.strftime('%Y-%m-%d')
        if ds not in daily:
            daily[ds] = {
                'temperatures': [],
                'feels_like': [],
                'humidity': [],
                'pressure': [],
                'wind_speed': [],
                'weather_main': item['weather'][0]['main']
            }
        daily[ds]['temperatures'].append(item['main']['temp'])
        daily[ds]['feels_like'].append(item['main']['feels_like'])
        daily[ds]['humidity'].append(item['main']['humidity'])
        daily[ds]['pressure'].append(item['main']['pressure'])
        daily[ds]['wind_speed'].append(item['wind']['speed']*3.6)
    return daily

@app.route('/predict_heatwave', methods=['POST'])
def predict_heatwave():
    try:
        city = request.json.get('city')
        if not city:
            return jsonify({'error': 'City required'}), 400

        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        r = requests.get(url)
        if r.status_code != 200:
            return jsonify({'error': 'City not found or API error'}), 404

        forecast_data = r.json()
        daily_data = process_daily_weather(forecast_data)
        predictions = []

        for ds, day in daily_data.items():
            max_temp = max(day['temperatures'])
            avg_feels = np.mean(day['feels_like'])
            avg_humidity = np.mean(day['humidity'])
            avg_pressure = np.mean(day['pressure'])
            avg_wind = np.mean(day['wind_speed'])

            # Create weather features for ML model (EXACT ORDER MATTERS!)
            weather_features = {
                'temp_c': max_temp,
                'feelslike_c': avg_feels,
                'humidity': avg_humidity,
                'dewpoint_c': max_temp - ((100 - avg_humidity) / 5),
                'pressure_mb': avg_pressure,
                'wind_kph': avg_wind,
                'hour': 14,
                'month': datetime.strptime(ds, '%Y-%m-%d').month,
                'temp_rolling_3h': max_temp,
                'temp_rolling_24h': max_temp,
                'daily_temp_range': max(day['temperatures']) - min(day['temperatures'])
            }

            prob = get_heatwave_percentage(weather_features)

            if prob >= 70: 
                risk="HIGH"
                color="#ff4444"
            elif prob >= 40: 
                risk="MEDIUM"
                color="#ff8800"
            else: 
                risk="LOW"
                color="#44ff44"

            predictions.append({
                'date': ds,
                'day_name': datetime.strptime(ds, '%Y-%m-%d').strftime('%A'),
                'max_temperature': round(max_temp,1),
                'feels_like': round(avg_feels,1),
                'humidity': round(avg_humidity),
                'weather': day['weather_main'],
                'heatwave_percentage': round(prob,1),
                'risk_level': risk,
                'color': color
            })

        return jsonify({
            'city': city,
            'predictions': predictions[:5],
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'API running!'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)



