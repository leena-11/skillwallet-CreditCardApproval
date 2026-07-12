import os
import numpy as np
import pandas as pd

def generate_credit_card_data(num_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Gender: Male (55%), Female (45%)
    gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.55, 0.45])
    
    # 2. Age: Uniformly distributed between 18 and 70
    age = np.random.randint(18, 71, size=num_samples)
    
    # 3. Education Level: Graduate (40%), Undergraduate (30%), Postgraduate (15%), High School (15%)
    education = np.random.choice(
        ['Graduate', 'Undergraduate', 'Postgraduate', 'High School'], 
        size=num_samples, 
        p=[0.40, 0.30, 0.15, 0.15]
    )
    
    # 4. Marital Status: Married (50%), Single (38%), Divorced (12%)
    marital_status = np.random.choice(
        ['Married', 'Single', 'Divorced'], 
        size=num_samples, 
        p=[0.50, 0.38, 0.12]
    )
    
    # 5. Employment Type: Salaried (60%), Self-Employed (25%), Student (10%), Unemployed (5%)
    employment_type = np.random.choice(
        ['Salaried', 'Self-Employed', 'Student', 'Unemployed'], 
        size=num_samples, 
        p=[0.60, 0.25, 0.10, 0.05]
    )
    
    # Adjust variables based on employment type
    years_in_job = []
    annual_income = []
    
    for i in range(num_samples):
        emp = employment_type[i]
        curr_age = age[i]
        
        # 6. Years in Current Job
        if emp in ['Student', 'Unemployed'] or curr_age <= 21:
            years = 0
        else:
            max_years = curr_age - 18
            years = int(np.random.beta(1.5, 3) * max_years)
            years = min(years, max_years)
        years_in_job.append(years)
        
        # 7. Annual Income (₹)
        if emp == 'Salaried':
            income = int(np.random.lognormal(mean=13.0, sigma=0.5))  # Mean around 450k-500k
        elif emp == 'Self-Employed':
            income = int(np.random.lognormal(mean=13.2, sigma=0.7))  # Higher variance
        elif emp == 'Student':
            income = np.random.randint(15000, 60000)  # Part-time / Pocket money
        else:  # Unemployed
            income = np.random.randint(0, 30000)
            
        # Bound income
        income = max(15000, min(income, 5000000))
        annual_income.append(income)
        
    years_in_job = np.array(years_in_job)
    annual_income = np.array(annual_income)
    
    # 8. Credit Score: Range 300 to 850
    # Center around 650
    credit_score = np.random.normal(loc=650, scale=100, size=num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # 9. Existing Loan Balance (₹)
    # Dependent on income and credit score
    existing_loan_balance = []
    for i in range(num_samples):
        inc = annual_income[i]
        cs = credit_score[i]
        
        # Low credit score or low income: less access to large loans
        # High credit score/income: higher borrowing capacity
        if cs < 500:
            debt_ratio = np.random.uniform(0.0, 0.2)
        else:
            debt_ratio = np.random.uniform(0.0, 1.5)
            
        balance = int(inc * debt_ratio)
        # Cap loan balance at 2M
        balance = min(balance, 2000000)
        existing_loan_balance.append(balance)
        
    existing_loan_balance = np.array(existing_loan_balance)
    
    # 10. Loan Amount Requested (₹)
    loan_amount_requested = []
    for i in range(num_samples):
        inc = annual_income[i]
        cs = credit_score[i]
        
        # People request loans ranging from small to large
        if cs > 700:
            request_ratio = np.random.uniform(0.1, 4.0)
        else:
            request_ratio = np.random.uniform(0.1, 1.5)
            
        req = int(inc * request_ratio)
        req = max(10000, min(req, 3000000))
        loan_amount_requested.append(req)
        
    loan_amount_requested = np.array(loan_amount_requested)
    
    # 11. Credit Card Inquiries: 0 to 10
    cc_inquiries = np.random.poisson(lam=1.5, size=num_samples)
    cc_inquiries = np.clip(cc_inquiries, 0, 10)
    
    # 12. Payment of Previous Dues
    # Salaried and high credit score -> higher chance of Paid on Time
    payment_dues = []
    for i in range(num_samples):
        cs = credit_score[i]
        prob_paid = 0.95 if cs >= 700 else (0.75 if cs >= 600 else 0.40)
        status = np.random.choice(['Paid on Time', 'Delayed'], p=[prob_paid, 1 - prob_paid])
        payment_dues.append(status)
        
    payment_dues = np.array(payment_dues)
    
    # --- Scoring Formula to determine Approval Status ---
    # Convert numerical variables to normalized scales for the scoring model
    norm_cs = (credit_score - 300) / 550.0  # 0 to 1
    norm_income = annual_income / 1000000.0  # ₹1M = 1.0
    
    # Debt service ratio (Existing Debt / Annual Income)
    debt_inc_ratio = existing_loan_balance / np.maximum(annual_income, 10000)
    # Requested Loan to Income ratio
    req_inc_ratio = loan_amount_requested / np.maximum(annual_income, 10000)
    
    # Scoring computation
    scores = []
    for i in range(num_samples):
        score = -1.0  # Base negative intercept
        
        # Credit score has the highest positive impact
        score += norm_cs[i] * 6.5
        
        # Income positive impact
        score += min(norm_income[i] * 1.5, 3.0)
        
        # High debt ratio reduces probability
        score -= min(debt_inc_ratio[i] * 2.0, 3.5)
        
        # High loan request ratio relative to income reduces probability
        score -= min(req_inc_ratio[i] * 1.5, 3.0)
        
        # Inquiries reduce score (higher inquiries = higher risk)
        score -= cc_inquiries[i] * 0.4
        
        # Payment of previous dues is critical
        if payment_dues[i] == 'Paid on Time':
            score += 2.2
        else:
            score -= 2.5
            
        # Employment Type impact
        emp = employment_type[i]
        if emp == 'Salaried':
            score += 1.2
        elif emp == 'Self-Employed':
            score += 0.6
        elif emp == 'Student':
            score -= 1.0
        elif emp == 'Unemployed':
            score -= 2.0
            
        # Experience/Years in job
        score += (years_in_job[i] / 10.0) * 0.4
        
        scores.append(score)
        
    scores = np.array(scores)
    
    # Add random standard normal noise to make it realistic
    noise = np.random.normal(loc=0.0, scale=0.5, size=num_samples)
    final_scores = scores + noise
    
    # Logistic probability
    probs = 1 / (1 + np.exp(-final_scores))
    
    # Assign target labels (1 = Approved, 0 = Rejected)
    approved = (probs >= 0.55).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Gender': gender,
        'Age': age,
        'Annual_Income': annual_income,
        'Education_Level': education,
        'Marital_Status': marital_status,
        'Employment_Type': employment_type,
        'Years_in_Current_Job': years_in_job,
        'Credit_Score': credit_score,
        'Existing_Loan_Balance': existing_loan_balance,
        'Loan_Amount_Requested': loan_amount_requested,
        'Credit_Card_Inquiries': cc_inquiries,
        'Payment_of_Previous_Dues': payment_dues,
        'Approved': approved
    })
    
    # Add small missing values for EDA demonstrating preprocessing (e.g. 1.5% random NaNs)
    # Only in specific fields to keep the dataset largely clean but provide realistic cleaning exercises.
    cols_with_nan = ['Credit_Score', 'Years_in_Current_Job', 'Marital_Status']
    for col in cols_with_nan:
        nan_indices = np.random.choice(num_samples, size=int(0.015 * num_samples), replace=False)
        df.loc[nan_indices, col] = np.nan
        
    return df

if __name__ == '__main__':
    dataset_dir = 'd:\\skillwallet\\CreditCardApproval\\dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    output_path = os.path.join(dataset_dir, 'credit_card_data.csv')
    print("Generating credit card approval dataset...")
    df = generate_credit_card_data()
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully! Saved to: {output_path}")
    print(f"Dataset Shape: {df.shape}")
    print(f"Class Distribution:\n{df['Approved'].value_counts(normalize=True)}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
