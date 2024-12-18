from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr, chi2_contingency
import dython
from dython.nominal import associations



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
    numerical_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    categorical_columns = [col for col in data.columns if pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col])]
    columns = data.columns.tolist()

    return render_template('index.html', 
                           numerical_columns=numerical_columns, 
                           categorical_columns=categorical_columns,
                           columns=columns)


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
            color_by = request_data.get('color_by')  # Optional

            # Validate columns
            if not x_column or not y_column:
                return jsonify({'error': 'Both x and y columns must be selected'}), 400
            if x_column not in data.columns or y_column not in data.columns:
                return jsonify({'error': 'Selected columns do not exist in the dataset'}), 400

            # Ensure x and y are numerical
            if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
                return jsonify({'error': 'x and y columns must be numerical'}), 400

            # Drop NaNs for selected columns
            scatter_data = data[[x_column, y_column, color_by]].dropna() if color_by else data[[x_column, y_column]].dropna()

            # Initialize plot
            plt.figure(figsize=(10, 6))

            # Handle coloring by a categorical variable
            if color_by and color_by in data.columns:
                # Ensure color_by is categorical
                if not pd.api.types.is_categorical_dtype(data[color_by]) and not pd.api.types.is_object_dtype(data[color_by]):
                    return jsonify({'error': 'Color-by column must be categorical'}), 400

                # Group by categories and plot
                unique_categories = scatter_data[color_by].unique()
                regressions = {}

                for category in unique_categories:
                    subset = scatter_data[scatter_data[color_by] == category]
                    x = subset[x_column].values.reshape(-1, 1)
                    y = subset[y_column].values

                    # Scatter points
                    plt.scatter(x, y, label=f'{category}', alpha=0.7)

                    # Linear regression for each category
                    model = LinearRegression()
                    model.fit(x, y)
                    y_pred = model.predict(x)
                    slope = model.coef_[0]
                    intercept = model.intercept_
                    regressions[category] = {
                        'slope': slope,
                        'intercept': intercept,
                        'equation': f'y = {slope:.2f}x + {intercept:.2f}'
                    }

                    # Plot regression line
                    plt.plot(x, y_pred, label=f'{category} (y = {slope:.2f}x + {intercept:.2f})')

                plt.legend()
            else:
                # Single scatterplot and regression
                x = scatter_data[x_column].values.reshape(-1, 1)
                y = scatter_data[y_column].values

                plt.scatter(x, y, color='blue', label='Data', alpha=0.7)

                # Linear regression
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                slope = model.coef_[0]
                intercept = model.intercept_

                plt.plot(x, y_pred, color='red', label=f'Regression (y = {slope:.2f}x + {intercept:.2f})')
                regressions = {
                    'overall': {
                        'slope': slope,
                        'intercept': intercept,
                        'equation': f'y = {slope:.2f}x + {intercept:.2f}'
                    }
                }

            # Finalize plot
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

            return jsonify({
                'plot': image_base64,
                'regressions': regressions
            })
        
        else:
            return jsonify({"error": "Analysis type not implemented"}), 400

    except Exception as e:
        print(f"Error: {e}")  # Log to console
        return jsonify({"error": str(e)}), 500


@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    data = request.get_json()  # Get the data from the frontend

    # Extract the relevant fields
    target_column = data.get('target_column')
    k_value = data.get('k')
    features = data.get('features')
    feature_values = data.get('feature_values')

    # Validation: Ensure no NaN or empty values
    if not target_column or not isinstance(k_value, int) or k_value <= 0:
        return jsonify({"error": "Invalid target column or k value"}), 400

    if not features or not feature_values or len(features) != len(feature_values):
        return jsonify({"error": "Features and their values must be provided correctly"}), 400

    # Check for NaN values in feature values
    if any(np.isnan(value) for value in feature_values):
        return jsonify({"error": "Feature values contain invalid (NaN) values"}), 400

    # Perform the prediction logic here (omitted for simplicity)
    prediction = "Some Prediction"  # Replace with actual model prediction logic
    class_probabilities = {"Class 1": 0.75, "Class 2": 0.25}  # Example probabilities

    return jsonify({
        "prediction": prediction,
        "probabilities": class_probabilities
    })
    

    

@app.route('/correlation_heatmap', methods=['POST'])
def correlation_heatmap():
    try:
        # Select only numerical columns
        numerical_data = data.select_dtypes(include=['number'])

        # Compute the correlation matrix
        corr_matrix = numerical_data.corr()

        # Create a heatmap without annotations
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')

        # Save the heatmap to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        # Encode the image as base64
        base64_image = base64.b64encode(buf.read()).decode('utf-8')

        return jsonify({'heatmap': base64_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



if __name__ == '__main__':
    app.run(debug=True)
