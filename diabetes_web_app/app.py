from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

app = Flask(__name__)


# Manual Logistic Regression - نفس الكلاس من الـ notebook
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_cost(self, y, y_pred, weights):
        m = len(y)
        epsilon = 1e-15
        cost = -1/m * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(weights ** 2)
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = np.array(y)

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            if self.regularization == 'l2':
                dw += (self.lambda_reg / m) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self.compute_cost(y, y_pred, self.weights)
            self.cost_history.append(cost)

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# تدريب الموديل - نفس إعدادات Final_Projects.ipynb مع SMOTE
def train_model():
    data = pd.read_csv('../diabetes.csv')

    # معالجة القيم الصفرية
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[zero_features] = data[zero_features].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    data[zero_features] = imputer.fit_transform(data[zero_features])

    # إزالة الـ Outliers باستخدام IQR
    numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    X = data[numerical_cols]
    y = data['Outcome']

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # تطبيع البيانات
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # موازنة الفئات - بديل SMOTE باستخدام oversampling بسيط
    from collections import Counter
    class_counts = Counter(y_train)
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    # Oversample minority class
    minority_indices = np.where(y_train.values == minority_class)[0]
    n_samples_needed = class_counts[majority_class] - class_counts[minority_class]
    
    if n_samples_needed > 0:
        np.random.seed(42)
        oversample_indices = np.random.choice(minority_indices, size=n_samples_needed, replace=True)
        X_train_balanced = np.vstack([X_train_scaled, X_train_scaled[oversample_indices]])
        y_train_balanced = np.concatenate([y_train.values, y_train.values[oversample_indices]])
    else:
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train.values

    # Manual Logistic Regression - أفضل موديل (80.6% Accuracy)
    model = LogisticRegressionManual(
        learning_rate=0.1,
        n_iterations=1000,
        regularization='l2',
        lambda_reg=0.01
    )
    model.fit(X_train_balanced, y_train_balanced)

    # حساب الدقة
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test.values)
    print(f"✓ Manual Logistic Regression Accuracy: {accuracy:.1%}")
    print(f"✓ Model: Best performing model from comparison")

    return model, scaler


# تحميل الموديل
model, scaler = train_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # التنبؤ
        probability = model.predict_proba(features_scaled)[0]
        prediction = 1 if probability >= 0.5 else 0

        result = {
            'prediction': 'مصاب بالسكري' if prediction == 1 else 'غير مصاب بالسكري',
            'probability': f'{probability*100:.1f}%',
            'risk_level': 'عالي' if probability > 0.7 else 'متوسط' if probability > 0.4 else 'منخفض',
            'is_diabetic': prediction == 1
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
