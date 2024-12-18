from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__, static_folder='static')

# Load dataset
data = pd.read_csv('data/clean_data.csv')
data.dropna(how='all', inplace=True)

# Plotting function
def generate_plot(x, y, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close()  # Prevent memory leak
    return image_base64





@app.route('/')
def index():
    # Get column names from the data
    columns = data.columns.tolist()  # List of column names
    return render_template('index.html', columns=columns)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        request_data = request.get_json()
        analysis_type = request_data.get('analysis_type')

        if analysis_type == 'summary':
            column = request_data.get('column')
            if not column:
                return jsonify({"error": "Column not provided"}), 400

            if column not in data.columns:
                return jsonify({"error": f"Column '{column}' does not exist in the dataset"}), 400
            
            # Calculate summary statistics
            summary_stats = data[column].describe()
            summary_stats_dict = summary_stats.to_dict()

            # Generate plots
            if pd.api.types.is_numeric_dtype(data[column]):
                plt.figure(figsize=(8, 6))
                plt.hist(data[column].dropna(), bins=10, color='blue', alpha=0.7)
                plt.title(f"Histogram of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
            else:
                plt.figure(figsize=(8, 6))
                data[column].value_counts().plot(kind='bar', color='blue', alpha=0.7)
                plt.title(f"Bar Chart of {column}")
                plt.xlabel(column)
                plt.ylabel("Count")

            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

            # Include plot in the response
            summary_stats_dict['plot'] = image_base64
            return jsonify(summary_stats_dict)
        
        elif analysis_type == 'scatterplot':
            x_column = request_data.get('x_column')
            y_column = request_data.get('y_column')

            if x_column is None or y_column is None:
                return jsonify({'error': 'Both x and y columns must be selected for the scatterplot'}), 400
            
            # Check if columns are numerical
            if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
                return jsonify({'error': 'Linear regression requires both columns to be numerical'}), 400

            # Prepare the data for regression
            x = data[x_column].dropna().values.reshape(-1, 1)
            y = data[y_column].dropna().values

            # Ensure matching lengths after dropping NaNs
            valid_idx = ~np.isnan(x.flatten()) & ~np.isnan(y)
            x = x[valid_idx]
            y = y[valid_idx]

            # Perform linear regression
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(x, y)

            # Generate scatterplot with regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Data')
            plt.plot(x, y_pred, color='red', label=f'Regression Line (y = {slope:.2f}x + {intercept:.2f})')
            plt.title(f'{x_column} vs {y_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.legend()

            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()

            # Return results including regression details
            return jsonify({
                'x_label': x_column,
                'y_label': y_column,
                'plot': image_base64,
                'equation': f'y = {slope:.2f}x + {intercept:.2f}',
                'r_squared': r_squared
            })

        else:
            return jsonify({"error": "Analysis type not implemented"}), 400

    except Exception as e:
        print(f"Error: {e}")  # Log to console
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
