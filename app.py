# app.py - Flask API Server (The Bridge Between Website and AI Model)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

# Import your model functions
from alzheimers_model import predict_progression, get_model_info

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow website to call this API

# ============ API ENDPOINTS ============

@app.route('/health', methods=['GET'])
def health_check():
    """Check if API and model are working"""
    return jsonify({
        'status': 'healthy',
        'message': 'Alzheimer\'s Progression API is running',
        'model_found': os.path.exists('alzheimers_model.pth')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - this is what your website calls"""
    try:
        # Get data from website
        data = request.get_json()
        print(f"Received prediction request: {data}")
        
        # Validate input
        if not data or 'patient_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing patient_data field'
            }), 400
        
        patient_data = data['patient_data']
        
        # Check if at least one year of data provided
        if not patient_data:
            return jsonify({
                'success': False,
                'error': 'No patient data provided'
            }), 400
        
        # Validate required fields for each year
        required_fields = ['age', 'bmi', 'education', 'mmse', 'cdr_sb', 'diagnosis', 
                          'apoe4', 'hypertension', 'gender']
        
        for year, year_data in patient_data.items():
            missing_fields = [field for field in required_fields if field not in year_data]
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': f'Missing fields for year {year}: {missing_fields}'
                }), 400
        
        # Check if model file exists
        if not os.path.exists('alzheimers_model.pth'):
            return jsonify({
                'success': False,
                'error': 'AI model not found. Please ensure alzheimers_model.pth is uploaded.'
            }), 500
        
        print(f"Making prediction for years: {list(patient_data.keys())}")
        
        # Make prediction using your AI model
        n_samples = data.get('n_samples', 50)  # Default 50 for speed
        results = predict_progression(patient_data, n_samples=n_samples)
        
        print(f"Prediction successful! Timeline: {len(results['timeline'])} points")
        
        # Return results to website
        return jsonify({
            'success': True,
            'predictions': results,
            'metadata': {
                'input_years': list(patient_data.keys()),
                'prediction_range': '0-10 years',
                'uncertainty_samples': n_samples,
                'model_type': 'Neural ODE with VAE'
            }
        })
        
    except Exception as e:
        # Log error for debugging
        error_msg = str(e)
        print(f"Prediction error: {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {error_msg}',
            'type': type(e).__name__
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the AI model"""
    try:
        if not os.path.exists('alzheimers_model.pth'):
            return jsonify({'error': 'Model file not found'}), 404
            
        info = get_model_info()
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Basic info about the API"""
    return jsonify({
        'name': 'Alzheimer\'s Progression Prediction API',
        'version': '1.0',
        'description': 'Neural ODE model for predicting CDRSB progression',
        'endpoints': {
            '/health': 'Check API status',
            '/predict': 'Make predictions (POST)',
            '/model-info': 'Get model information'
        },
        'model_ready': os.path.exists('alzheimers_model.pth')
    })

# ============ STARTUP CHECK ============

def startup_check():
    """Check if everything is ready when server starts"""
    print("="*50)
    print("ALZHEIMER'S PROGRESSION API STARTING...")
    print("="*50)
    
    if os.path.exists('alzheimers_model.pth'):
        try:
            info = get_model_info()
            print(f"✅ Model loaded successfully!")
            print(f"📁 Model size: {info.get('file_size_mb', 'unknown')} MB")
            print(f"🧠 Parameters: {info.get('total_parameters', 'unknown'):,}")
            print(f"📊 Static features: {info.get('static_dim', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Model found but failed to load: {e}")
    else:
        print("❌ WARNING: alzheimers_model.pth not found!")
        print("   The API will start but predictions will fail.")
        print("   Please upload your trained model file.")
    
    print(f"🌐 API will be available at: http://localhost:5000")
    print("="*50)

# ============ RUN SERVER ============

if __name__ == '__main__':
    # Run startup check
    startup_check()
    
    # Start Flask server
    # For production, use: gunicorn app:app
    # For local testing, use: python app.py
    app.run(
        debug=True,      # Shows detailed errors (disable in production)
        host='0.0.0.0',  # Allow external connections
        port=5000        # Port number
    )
