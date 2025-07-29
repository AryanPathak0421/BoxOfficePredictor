from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib

# ✅ Import your custom model class
from predictor import TicketSalesPredictor


# Constants
MODEL_PATH = "ticket_sales_predictor.joblib"

# Initialize Flask app
app = Flask(__name__)

# ✅ Enable CORS for only your frontend domain
CORS(app, resources={r"/predict": {"origins": "https://boxofficepredictor.netlify.app"}})

# Load the trained model
try:
    predictor = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if not request.is_json:
        return jsonify({'error': 'Missing or invalid JSON body'}), 400

    try:
        movie_data = request.get_json()
        tickets = predictor.predict_ticket_sales(movie_data)

        return jsonify({
            'predicted_ticket_sales': int(tickets)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5040)




    # # api.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # ✅ Import CORS
# import joblib

# # ✅ Import your custom model class
# from predictor import TicketSalesPredictor

# # Constants
# MODEL_PATH = "ticket_sales_predictor.joblib"

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # ✅ Enable CORS for all domains (you can restrict it if needed)

# # Load the trained model
# try:
#     predictor = joblib.load(MODEL_PATH)
#     print(f"✅ Model loaded from {MODEL_PATH}")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     predictor = None

# # Prediction endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     if predictor is None:
#         return jsonify({'error': 'Model not loaded'}), 500

#     if not request.is_json:
#         return jsonify({'error': 'Missing or invalid JSON body'}), 400

#     try:
#         movie_data = request.get_json()
#         tickets = predictor.predict_ticket_sales(movie_data)

#         return jsonify({
#             'predicted_ticket_sales': int(tickets)
#         })

#     except Exception as e:
#         return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# # Run the server
# if __name__ == '__main__':
#     # Use a different port if 5500 is occupied
#     app.run(host='0.0.0.0', port=5040)

# api.py
