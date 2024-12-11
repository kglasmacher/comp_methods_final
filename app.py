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

        elif analysis_type == 'scatter_plot':
            x_col = request_data.get('x_column')
            y_col = request_data.get('y_column')
            if x_col not in data.columns or y_col not in data.columns:
                raise ValueError(f"Invalid columns: {x_col}, {y_col}")
            image = generate_plot(data[x_col], data[y_col])
            return jsonify({"image": image})

        else:
            return jsonify({"error": "Analysis type not implemented"}), 400

    except Exception as e:
        print(f"Error: {e}")  # Log to console
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
