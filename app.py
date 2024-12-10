from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

data = pd.read_csv('data/dataset.csv') # load preprocessed dataframe

# API routes
@app.route('/')
def home():
    #return "<h1>Flask App is Running</h1>" # for debugging
    return render_template('index.html') # front end html file

@app.route('/analyze', methods=['POST'])
def analyze():
    # Parse request data
    request_data = request.get_json()
    analysis_type = request_data.get('analysis_type')
    parameter = request_data.get('parameter', None)

    # Perform analysis based on type
    if analysis_type == 'summary':
        result = data.describe().to_dict()
        return jsonify(result)

    elif analysis_type == 'scatter_plot':
        x_column = request_data.get('x_column')
        y_column = request_data.get('y_column')
        img = create_scatter_plot(data, x_column, y_column)
        return jsonify({'image': img})

    elif analysis_type == 'filter':
        threshold = float(parameter)
        filtered_data = data[data['some_column'] > threshold]
        result = filtered_data.to_dict(orient='records')
        return jsonify(result)

    return jsonify({'error': 'Invalid analysis type'}), 400

def create_scatter_plot(df, x_column, y_column):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.6)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Scatter Plot: {x_column} vs {y_column}')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return img_base64

@app.route('/columns', methods=['GET'])
def get_columns():
    columns = data.columns.tolist()
    return jsonify(columns)

if __name__ == '__main__':
    app.run(debug=True)
