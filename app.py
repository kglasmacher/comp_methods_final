from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

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
        print(f"Request data received: {request_data}")  # For debugging

        analysis_type = request_data.get('analysis_type')

        if analysis_type == 'summary':
            column = request_data.get('column')
            if not column:
                return jsonify({"error": "Column not provided"}), 400

            # Check if the column exists in the DataFrame
            if column not in data.columns:
                return jsonify({"error": f"Column '{column}' does not exist in the dataset"}), 400
            
            # Calculate summary statistics
            summary_stats = data[column].describe()
            summary_stats_dict = summary_stats.to_dict()

            # Return the summary stats as JSON
            return jsonify(summary_stats_dict)

        elif analysis_type == 'scatterplot':
            # Get the x and y columns for the scatterplot
            x_column = request_data.get('x_column')
            y_column = request_data.get('y_column')

            if x_column is None or y_column is None:
                return jsonify({'error': 'Both x and y columns must be selected for the scatterplot'}), 400
            
            # Prepare scatterplot data
            scatter_data = {
                'x': data[x_column].tolist(),
                'y': data[y_column].tolist(),
                'x_label': x_column,
                'y_label': y_column
            }

            return jsonify(scatter_data)

        else:
            return jsonify({"error": "Analysis type not implemented"}), 400

    except Exception as e:
        print(f"Error: {e}")  # Log to console
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
