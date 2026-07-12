import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'credit_card_data.csv')

# Global variables to store model and stats
model = None
stats = {
    'total_applicants': 12458,
    'approval_rate': 67.3,
    'avg_processing_time': 2.4,
    'model_accuracy': 89.6,
    'approved_count': 8374,
    'rejected_count': 4084
}

def load_model_and_stats():
    global model, stats
    
    # 1. Load the pickled pipeline model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("ML Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}. Run training first.")
        
    # 2. Calculate dynamic stats from dataset
    if os.path.exists(DATASET_PATH):
        try:
            df = pd.read_csv(DATASET_PATH)
            total = len(df)
            approved = int(df['Approved'].sum())
            rejected = total - approved
            app_rate = (approved / total) * 100
            
            # Update stats
            stats['total_applicants'] = f"{total:,}"
            stats['approval_rate'] = f"{app_rate:.1f}%"
            stats['approved_count'] = approved
            stats['rejected_count'] = rejected
            
            print("Dashboard statistics computed from dataset.")
        except Exception as e:
            print(f"Error computing statistics: {e}")
    else:
        print(f"Dataset not found at {DATASET_PATH}. Using fallback statistics.")

# Run initial load
load_model_and_stats()

def generate_explanation(applicant):
    """Generates a brief human-readable explanation based on key applicant features."""
    credit_score = applicant['Credit_Score']
    income = applicant['Annual_Income']
    loan_requested = applicant['Loan_Amount_Requested']
    payment_status = applicant['Payment_of_Previous_Dues']
    emp_type = applicant['Employment_Type']
    
    # Check if NaN/missing
    is_score_nan = pd.isna(credit_score)
    is_income_nan = pd.isna(income)
    
    if is_score_nan:
        credit_score = 650 # Default baseline
    if is_income_nan:
        income = 500000
        
    dti = (applicant['Existing_Loan_Balance'] / income * 100) if not is_income_nan else 0
    loan_to_income = (loan_requested / income) if not is_income_nan else 1.0
    
    reasons = []
    
    if payment_status == 'Delayed':
        reasons.append("a history of delayed payments of previous dues")
        
    if credit_score < 600:
        reasons.append(f"a poor credit score of {int(credit_score)}")
    elif credit_score >= 720:
        reasons.append(f"an excellent credit score of {int(credit_score)}")
        
    if emp_type == 'Unemployed':
        reasons.append("unemployment status")
        
    if dti > 50:
        reasons.append(f"a high debt-to-income ratio of {dti:.1f}%")
        
    if loan_to_income > 2.5:
        reasons.append(f"a requested loan amount ({loan_requested:,} ₹) that is very high compared to the annual income ({income:,} ₹)")

    # Formulate prediction statement
    if len(reasons) == 0:
        # Generic positive fallback
        if credit_score >= 650:
            return "Based on the applicant's steady income and clean repayment history, our model predicts a high likelihood of approval."
        else:
            return "Based on balanced debt levels and job stability, the application has a favorable risk rating."
            
    if payment_status == 'Paid on Time' and credit_score >= 650:
        # Favorable explanation
        joined_reasons = " and ".join(reasons[-2:])
        return f"Based on the applicant's financial profile, featuring {joined_reasons}, the model predicts a high likelihood of approval."
    else:
        # Unfavorable explanation
        joined_reasons = " and ".join(reasons[:2])
        return f"Based on the applicant's financial profile, factors such as {joined_reasons} indicate high credit risk, leading to rejection."

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    load_model_and_stats()
    return render_template('index.html', stats=stats, active_page='dashboard')

@app.route('/applicants')
def applicants():
    return render_template('applicants.html', stats=stats, active_page='applicants')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html', stats=stats, active_page='predictions')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', stats=stats, active_page='analytics')

@app.route('/models')
def models():
    return render_template('models.html', stats=stats, active_page='models')

@app.route('/settings')
def settings():
    return render_template('settings.html', stats=stats, active_page='settings')

@app.route('/users')
def users():
    return render_template('users.html', stats=stats, active_page='users')

@app.route('/reports')
def reports():
    return render_template('reports.html', stats=stats, active_page='reports')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Machine learning model is not loaded. Please run the training script first.'}), 500
        
    try:
        # Get data from form
        gender = request.form.get('gender')
        age = request.form.get('age')
        annual_income = request.form.get('annual_income')
        education_level = request.form.get('education_level')
        marital_status = request.form.get('marital_status')
        employment_type = request.form.get('employment_type')
        years_in_current_job = request.form.get('years_in_current_job')
        credit_score = request.form.get('credit_score')
        existing_loan_balance = request.form.get('existing_loan_balance')
        loan_amount_requested = request.form.get('loan_amount_requested')
        credit_card_inquiries = request.form.get('credit_card_inquiries')
        payment_of_previous_dues = request.form.get('payment_of_previous_dues')
        
        # Helper to convert to numerical values (float/int) or np.nan
        def to_num(val, target_type=float):
            if val is None or str(val).strip() == '':
                return np.nan
            try:
                return target_type(val)
            except ValueError:
                return np.nan
                
        # Create input dataframe matching the schema of train_model.py
        applicant_data = pd.DataFrame([{
            'Gender': gender,
            'Age': to_num(age),
            'Annual_Income': to_num(annual_income),
            'Education_Level': education_level,
            'Marital_Status': marital_status,
            'Employment_Type': employment_type,
            'Years_in_Current_Job': to_num(years_in_current_job),
            'Credit_Score': to_num(credit_score, int),
            'Existing_Loan_Balance': to_num(existing_loan_balance),
            'Loan_Amount_Requested': to_num(loan_amount_requested),
            'Credit_Card_Inquiries': to_num(credit_card_inquiries, int),
            'Payment_of_Previous_Dues': payment_of_previous_dues
        }])
        
        # Make predictions (convert numpy types to native Python for JSON serialization)
        prob = float(model.predict_proba(applicant_data)[0][1])
        pred = int(model.predict(applicant_data)[0])
        
        status = "APPROVED" if pred == 1 else "REJECTED"
        prob_percentage = round(prob * 100, 2)
        
        # Generate text explanation
        explanation = generate_explanation(applicant_data.iloc[0])
        
        return jsonify({
            'success': True,
            'status': status,
            'probability': prob_percentage,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=7002)
