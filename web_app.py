from flask import Flask, render_template
import boto3
from datetime import datetime
import json

app = Flask(__name__)

# Initialize AWS DynamoDB client
dynamodb = boto3.resource('dynamodb')
table_name = "BodyFatPredictions"
table = dynamodb.Table(table_name)

@app.route('/')
def index():
    try:
        # Scan DynamoDB table to get all predictions
        response = table.scan()
        items = response.get('Items', [])
        
        # Extract unique names
        names = sorted(set(item.get('name', 'N/A') for item in items if item.get('name')))
        
        # Group predictions by name
        predictions_by_name = {}
        for item in items:
            name = item.get('name', 'N/A')
            if name not in predictions_by_name:
                predictions_by_name[name] = []
            predictions_by_name[name].append({
                'prediction_id': item['prediction_id'],
                'name': name,
                'height_cm': float(item['height_cm']),
                'weight_kg': float(item['weight_kg']),
                'predicted_body_fat': float(item['predicted_body_fat']),
                'category': item['category'],
                'timestamp': item['timestamp'],
            })
        
        # Sort predictions by timestamp for each name
        for name in predictions_by_name:
            predictions_by_name[name].sort(key=lambda x: x['timestamp'])
        
        # Prepare chart data
        chart_data = {}
        for name in predictions_by_name:
            timestamps = [p['timestamp'] for p in predictions_by_name[name]]
            body_fats = [p['predicted_body_fat'] for p in predictions_by_name[name]]
            chart_data[name] = {
                'labels': timestamps,
                'data': body_fats
            }
        
        return render_template('index.html', names=names, predictions_by_name=predictions_by_name, chart_data=json.dumps(chart_data))
    except Exception as e:
        return f"Error loading predictions: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
