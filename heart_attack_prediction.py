import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ============================================
# HEART ATTACK DETECTION MODEL
# ============================================

class HeartAttackDetectionModel:
    """
    A machine learning model for heart attack prediction based on medical features.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        Load or generate heart attack dataset.
        Dataset features:
        - age: Age of the patient
        - sex: Gender (0=female, 1=male)
        - cp: Chest pain type (0-3)
        - trestbps: Resting blood pressure
        - chol: Serum cholesterol
        - fbs: Fasting blood sugar (>120 mg/dl: 1=true, 0=false)
        - restecg: Resting electrocardiographic results (0-2)
        - thalach: Maximum heart rate achieved
        - exang: Exercise induced angina (1=yes, 0=no)
        - oldpeak: ST depression induced by exercise
        - slope: Slope of the ST segment (0-2)
        - ca: Number of major vessels (0-3)
        - thal: Thalassemia (0=normal, 1=fixed defect, 2=reversible defect)
        - target: Heart attack (1=yes, 0=no)
        """
        # Generate synthetic dataset for demonstration
        np.random.seed(42)
        n_samples = 300
        
        data = {
            'age': np.random.randint(29, 77, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(126, 565, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(60, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples),
        }
        
        X = pd.DataFrame(data)
        # Create target with some correlation to features
        y = ((X['age'] > 50) & (X['chol'] > 240) | (X['thalach'] < 100) & (X['exang'] == 1)).astype(int)
        y = y + np.random.binomial(1, 0.2, n_samples)  # Add some noise
        y = np.clip(y, 0, 1)
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split and scale the data"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train_models(self):
        """Train multiple models and select the best one"""
        print("🏥 Training Heart Attack Detection Models...\n")
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        lr_score = lr_model.score(self.X_test, self.y_test)
        print(f"Logistic Regression Accuracy: {lr_score:.4f}")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(self.X_train, self.y_train)
        rf_score = rf_model.score(self.X_test, self.y_test)
        print(f"Random Forest Accuracy: {rf_score:.4f}")
        
        # Select the best model
        if rf_score >= lr_score:
            self.model = rf_model
            print(f"\n✓ Random Forest selected as the best model\n")
        else:
            self.model = lr_model
            print(f"\n✓ Logistic Regression selected as the best model\n")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        print("=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        print(f"Accuracy:  {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred):.4f}")
        print(f"Recall:    {recall_score(self.y_test, y_pred):.4f}")
        print(f"F1-Score:  {f1_score(self.y_test, y_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Heart Attack', 'Heart Attack']))
    
    def predict_risk(self, patient_data):
        """
        Predict heart attack risk for a new patient.
        
        Args:
            patient_data: dict with patient features
            
        Returns:
            Risk level (0=Low, 1=High) and probability
        """
        # Convert to DataFrame and scale
        df = pd.DataFrame([patient_data])
        scaled_data = self.scaler.transform(df)
        
        prediction = self.model.predict(scaled_data)[0]
        probability = self.model.predict_proba(scaled_data)[0, 1]
        
        risk_level = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
        sex_label = "Male" if patient_data['sex'] == 1 else "Female"
        
        return {
            'risk_level': risk_level,
            'prediction': prediction,
            'probability': f"{probability * 100:.2f}%",
            'sex': sex_label
        }


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Initialize model
    model = HeartAttackDetectionModel()
    
    # Load data
    X, y = model.load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Prepare data
    model.prepare_data(X, y)
    
    # Train models
    model.train_models()
    
    # Evaluate
    model.evaluate_model()
    
    # Example prediction
    print("\n" + "=" * 50)
    print("EXAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Sample patient 1: High risk
    patient_1 = {
        'age': 65,
        'sex': 1,
        'cp': 3,
        'trestbps': 160,
        'chol': 290,
        'fbs': 1,
        'restecg': 1,
        'thalach': 85,
        'exang': 1,
        'oldpeak': 2.5,
        'slope': 2,
        'ca': 2,
        'thal': 2
    }
    
    # Sample patient 2: Low risk
    patient_2 = {
        'age': 29,
        'sex': 1,
        'cp': 0,
        'trestbps': 100,
        'chol': 200,
        'fbs': 0,
        'restecg': 0,
        'thalach': 110,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 0,
        'ca': 0,
        'thal': 0
    }
    
    result_1 = model.predict_risk(patient_1)
    result_2 = model.predict_risk(patient_2)
    
    print(f"\nPatient 1 ({patient_1['age']} year-old, {result_1['sex']}):")
    print(f"  Risk Level: {result_1['risk_level']}")
    print(f"  Probability: {result_1['probability']}")
    
    print(f"\nPatient 2: ({patient_2['age']} year-old, {result_2['sex']}):")
    print(f"  Risk Level: {result_2['risk_level']}")
    print(f"  Probability: {result_2['probability']}")