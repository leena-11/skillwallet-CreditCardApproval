# Credit Card Approval Prediction System

An end-to-end Machine Learning project that predicts the likelihood of credit card approval based on applicant financial profile and credit history. It features a complete ML training pipeline and a premium, responsive Flask dashboard interface with visual analytics.

## 📊 Model Performance Summary

During the model training phase, four machine learning algorithms were trained and compared on the dataset (split 80:20):

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **94.40%** | **97.48%** | **94.40%** | **95.92%** |
| Logistic Regression | 93.10% | 97.01% | 92.97% | 94.95% |
| Random Forest Classifier | 92.50% | 97.41% | 91.68% | 94.46% |
| Decision Tree Classifier | 90.90% | 96.47% | 90.24% | 93.25% |

**XGBoost Classifier** was selected as the best-performing model (F1-score: **95.92%**) and serialized into `model.pkl`.

---

## 📁 Project Directory Structure

```text
CreditCardApproval/
│
├── dataset/
│   ├── credit_card_data.csv        # Generated dataset (5,000 records)
│   └── generate_dataset.py         # Synthetic dataset generator script
│
├── notebooks/
│   ├── eda_and_modeling.ipynb      # Jupyter Notebook for EDA & model testing
│   └── generate_notebook.py        # Programmatic notebook generator script
│
├── static/
│   └── css/
│       └── style.css               # Modern glassmorphism dashboard styling
│
├── templates/
│   └── index.html                  # Jinja2 HTML dashboard template
│
├── app.py                          # Flask web application server
├── train_model.py                  # Production-ready ML training pipeline script
├── model.pkl                       # Serialized best XGBoost pipeline (preprocessor + model)
├── requirements.txt                # Project python packages list
└── README.md                       # Documentation file
```

---

## 🛠️ Technologies Used

- **Data & Preprocessing**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: XGBoost, Scikit-learn
- **Visualization**: Matplotlib, Seaborn (EDA), Chart.js (Dashboard charts)
- **Web App Framework**: Flask, Jinja2, Gunicorn
- **Frontend & Styling**: Vanilla HTML5, Custom CSS3, FontAwesome Icons

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally:

### 1. Clone or Move to Workspace
Open your terminal and navigate to the directory containing the project:
```bash
cd CreditCardApproval
```

### 2. Create and Activate Virtual Environment
It is highly recommended to isolate dependencies in a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python packages:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Execute the Project

### Step 1: Generate the Dataset (Optional)
A pre-generated dataset with 5,000 records is already provided. If you want to re-generate the dataset with custom settings, run:
```bash
python dataset/generate_dataset.py
```

### Step 2: Train the Models
Train the classifiers, evaluate their performance, select the best model, and save the model to `model.pkl`:
```bash
python train_model.py
```

### Step 3: Start the Flask Web Application
Run the web application server:
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 🖥️ Application Features

1. **Analytical Dashboard Overview**:
   - Displays KPIs like Total Applicants, Approval Rate, and Model Accuracy.
   - Shows interactive analytics: **Approval Rate Trend** (Line Chart) and **Approval Distribution** (Donut Chart).
2. **Real-time Prediction**:
   - Takes 12 financial and personal inputs including Age, Annual Income, Credit Score, Debt Balance, and Previous Dues.
   - Automatically handles missing or blank inputs using the built-in preprocessor pipeline imputer.
3. **Intelligent Prediction Outcomes**:
   - Renders color-coded status badges (**APPROVED** in green / **REJECTED** in red).
   - Computes prediction confidence probability.
   - Generates dynamically tailored natural language explanations based on the input values.
