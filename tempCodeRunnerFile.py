from flask import Flask, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('best_model.keras')

# Get model summary as a string
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
model_summary_text = "\n".join(model_summary)

# Create dummy input data
input_data = np.random.rand(1, 224, 224, 3)  # Assuming an image of size 224x224 with 3 color channels

# Create intermediate layer model
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('block1_conv1').output)

# Get intermediate outputs
intermediate_output = intermediate_layer_model(input_data)

# Convert intermediate output to string
intermediate_output_text = str(intermediate_output.numpy())

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    # Enhanced HTML template with CSS for centering, styling, and scrolling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
            }
            .container {
                text-align: center;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                width: 90%;
                height: 90vh;
                overflow-y: auto; /* Allows vertical scrolling */
                overflow-x: auto; /* Allows horizontal scrolling */
            }
            pre {
                text-align: left;
                background-color: #e8e8e8;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto; /* Ensures horizontal scrolling for wide content */
                white-space: pre-wrap; /* Ensures wrapping of content */
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Summary</h1>
            <pre>{{ model_summary }}</pre>
            <h1>Intermediate Output</h1>
            <pre>{{ intermediate_output }}</pre>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, model_summary=model_summary_text, intermediate_output=intermediate_output_text)

if __name__ == '__main__':
    app.run(debug=True)
