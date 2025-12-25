from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from collections import Counter
import uvicorn


# Manual Logistic Regression
class LogisticRegressionManual:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        y = np.array(y)

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            if self.regularization == 'l2':
                dw += (self.lambda_reg / m) * self.weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)


def train_model():
    data = pd.read_csv('diabetes.csv')
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[zero_features] = data[zero_features].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    data[zero_features] = imputer.fit_transform(data[zero_features])

    numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]

    X = data[numerical_cols]
    y = data['Outcome']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    class_counts = Counter(y_train)
    majority = max(class_counts, key=class_counts.get)
    minority = min(class_counts, key=class_counts.get)
    minority_idx = np.where(y_train.values == minority)[0]
    n_needed = class_counts[majority] - class_counts[minority]

    if n_needed > 0:
        np.random.seed(42)
        extra_idx = np.random.choice(minority_idx, size=n_needed, replace=True)
        X_balanced = np.vstack([X_train_scaled, X_train_scaled[extra_idx]])
        y_balanced = np.concatenate([y_train.values, y_train.values[extra_idx]])
    else:
        X_balanced, y_balanced = X_train_scaled, y_train.values

    model = LogisticRegressionManual(learning_rate=0.1, n_iterations=1000, lambda_reg=0.01)
    model.fit(X_balanced, y_balanced)
    return model, scaler


model, scaler = train_model()
app = FastAPI()


INDEX_HTML = '''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تشخيص مرض السكري</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Cairo', sans-serif; }
        body { background: #f8f9fa; min-height: 100vh; }
        .navbar { background: #fff; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
        .main-card { background: #fff; border: none; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        .card-header-custom { background: #fff; border-bottom: 1px solid #eee; padding: 24px 30px; border-radius: 16px 16px 0 0; }
        .card-header-custom h2 { color: #1a1a2e; font-weight: 700; }
        .form-label { color: #333; font-weight: 600; margin-bottom: 8px; }
        .form-control { border: 2px solid #e9ecef; border-radius: 10px; padding: 12px 16px; transition: all 0.3s; }
        .form-control:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
        .input-group-text { background: #f8f9fa; border: 2px solid #e9ecef; border-right: none; color: #6c757d; border-radius: 0 10px 10px 0; }
        .input-group .form-control { border-radius: 10px 0 0 10px; }
        .btn-primary-custom { background: #3b82f6; border: none; border-radius: 10px; padding: 14px 48px; font-size: 16px; font-weight: 600; transition: all 0.3s; }
        .btn-primary-custom:hover { background: #2563eb; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
        .feature-info { font-size: 12px; color: #9ca3af; margin-top: 4px; }
        .input-card { background: #fafafa; border-radius: 12px; padding: 16px; margin-bottom: 16px; border: 1px solid #f0f0f0; }
        .icon-circle { width: 48px; height: 48px; background: #eff6ff; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; margin-left: 12px; }
        .footer-note { color: #9ca3af; font-size: 13px; }
    </style>
</head>
<body>
    <nav class="navbar py-3 mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h4"><span style="font-size: 28px;">🩺</span> نظام التنبؤ بمرض السكري</span>
            <span class="badge bg-primary">دقة الموديل: 80.6%</span>
        </div>
    </nav>
    <div class="container pb-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-card">
                    <div class="card-header-custom">
                        <h2 class="mb-1">إدخال البيانات الصحية</h2>
                        <p class="text-muted mb-0">أدخل المؤشرات الصحية للحصول على تقييم احتمالية الإصابة بالسكري</p>
                    </div>
                    <div class="card-body p-4">
                        <form action="/predict" method="POST">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">🤰</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">عدد مرات الحمل</label>
                                                <input type="number" class="form-control" name="pregnancies" min="0" max="20" value="0" required>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">🩸</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">مستوى الجلوكوز</label>
                                                <div class="input-group">
                                                    <input type="number" class="form-control" name="glucose" min="0" max="300" value="120" required>
                                                    <span class="input-group-text">mg/dL</span>
                                                </div>
                                                <small class="feature-info">الطبيعي: 70-100 | ما قبل السكري: 100-125</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">💓</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">ضغط الدم</label>
                                                <div class="input-group">
                                                    <input type="number" class="form-control" name="blood_pressure" min="0" max="200" value="70" required>
                                                    <span class="input-group-text">mm Hg</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">📏</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">سمك الجلد</label>
                                                <div class="input-group">
                                                    <input type="number" class="form-control" name="skin_thickness" min="0" max="100" value="20" required>
                                                    <span class="input-group-text">mm</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">💉</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">الأنسولين</label>
                                                <div class="input-group">
                                                    <input type="number" class="form-control" name="insulin" min="0" max="900" value="80" required>
                                                    <span class="input-group-text">mu U/ml</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">⚖️</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">مؤشر كتلة الجسم (BMI)</label>
                                                <div class="input-group">
                                                    <input type="number" step="0.1" class="form-control" name="bmi" min="0" max="70" value="25" required>
                                                    <span class="input-group-text">kg/m²</span>
                                                </div>
                                                <small class="feature-info">الطبيعي: 18.5-25 | زيادة الوزن: 25-30</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">👨‍👩‍👧</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">دالة النسب السكري</label>
                                                <input type="number" step="0.001" class="form-control" name="dpf" min="0" max="3" value="0.5" required>
                                                <small class="feature-info">التاريخ العائلي للسكري (0.0 - 2.5)</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="input-card">
                                        <div class="d-flex align-items-start">
                                            <div class="icon-circle">🎂</div>
                                            <div class="flex-grow-1">
                                                <label class="form-label">العمر</label>
                                                <div class="input-group">
                                                    <input type="number" class="form-control" name="age" min="1" max="120" value="30" required>
                                                    <span class="input-group-text">سنة</span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="text-center mt-4 pt-2">
                                <button type="submit" class="btn btn-primary-custom btn-lg text-white">تحليل البيانات</button>
                            </div>
                        </form>
                    </div>
                </div>
                <div class="text-center mt-4"><p class="footer-note">⚠️ هذا النظام للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب</p></div>
            </div>
        </div>
    </div>
</body>
</html>'''


RESULT_HTML = '''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>نتيجة التشخيص</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ font-family: 'Cairo', sans-serif; }}
        body {{ background: #f8f9fa; min-height: 100vh; }}
        .navbar {{ background: #fff; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .main-card {{ background: #fff; border: none; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); overflow: hidden; }}
        .result-header {{ padding: 40px; text-align: center; }}
        .result-positive {{ background: #fef2f2; border-bottom: 3px solid #ef4444; }}
        .result-negative {{ background: #f0fdf4; border-bottom: 3px solid #22c55e; }}
        .result-icon {{ width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 40px; margin: 0 auto 16px; }}
        .icon-positive {{ background: #fee2e2; }}
        .icon-negative {{ background: #dcfce7; }}
        .result-title {{ font-size: 24px; font-weight: 700; margin-bottom: 0; }}
        .text-positive {{ color: #dc2626; }}
        .text-negative {{ color: #16a34a; }}
        .stats-card {{ background: #f8f9fa; border-radius: 12px; padding: 20px; text-align: center; }}
        .stats-value {{ font-size: 32px; font-weight: 700; color: #1a1a2e; }}
        .stats-label {{ color: #6b7280; font-size: 14px; }}
        .progress-custom {{ height: 12px; border-radius: 6px; background: #e5e7eb; }}
        .risk-badge {{ padding: 6px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; }}
        .risk-high {{ background: #fee2e2; color: #dc2626; }}
        .risk-medium {{ background: #fef3c7; color: #d97706; }}
        .risk-low {{ background: #dcfce7; color: #16a34a; }}
        .recommendation-card {{ background: #f8f9fa; border-radius: 12px; padding: 24px; }}
        .btn-outline-custom {{ border: 2px solid #e5e7eb; border-radius: 10px; padding: 12px 32px; font-weight: 600; color: #374151; }}
        .btn-outline-custom:hover {{ background: #f3f4f6; }}
    </style>
</head>
<body>
    <nav class="navbar py-3 mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h4"><span style="font-size: 28px;">🩺</span> نظام التنبؤ بمرض السكري</span>
        </div>
    </nav>
    <div class="container pb-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="main-card">
                    <div class="result-header {header_class}">
                        <div class="result-icon {icon_class}">{icon}</div>
                        <h2 class="result-title {text_class}">{prediction}</h2>
                    </div>
                    <div class="card-body p-4">
                        <div class="row g-3 mb-4">
                            <div class="col-md-6">
                                <div class="stats-card">
                                    <div class="stats-value">{probability}</div>
                                    <div class="stats-label">احتمالية الإصابة</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="stats-card">
                                    <span class="risk-badge {risk_class}">{risk_level}</span>
                                    <div class="stats-label mt-2">مستوى الخطورة</div>
                                </div>
                            </div>
                        </div>
                        <div class="mb-4">
                            <div class="d-flex justify-content-between mb-2">
                                <span class="text-muted">مؤشر الخطورة</span>
                                <span class="fw-bold">{probability}</span>
                            </div>
                            <div class="progress progress-custom">
                                <div class="progress-bar {progress_class}" style="width: {probability}"></div>
                            </div>
                        </div>
                        <div class="recommendation-card">
                            <h5 class="{rec_title_class}">📋 {rec_title}</h5>
                            <ul>{recommendations}</ul>
                        </div>
                        <div class="text-center mt-4">
                            <a href="/" class="btn btn-outline-custom">← فحص جديد</a>
                        </div>
                    </div>
                </div>
                <div class="text-center mt-4"><p style="color: #9ca3af; font-size: 13px;">⚠️ هذا النظام للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب</p></div>
            </div>
        </div>
    </div>
</body>
</html>'''


@app.get("/", response_class=HTMLResponse)
async def home():
    return INDEX_HTML


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    pregnancies: float = Form(...),
    glucose: float = Form(...),
    blood_pressure: float = Form(...),
    skin_thickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    dpf: float = Form(...),
    age: float = Form(...)
):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0]
    is_diabetic = probability >= 0.5

    if probability > 0.7:
        risk_level, risk_class = "عالي", "risk-high"
        progress_class = "bg-danger"
    elif probability > 0.4:
        risk_level, risk_class = "متوسط", "risk-medium"
        progress_class = "bg-warning"
    else:
        risk_level, risk_class = "منخفض", "risk-low"
        progress_class = "bg-success"

    if is_diabetic:
        recommendations = """
            <li>استشر طبيبك في أقرب وقت ممكن للتأكد من التشخيص</li>
            <li>قم بإجراء فحص HbA1c (السكر التراكمي)</li>
            <li>راقب مستوى السكر في الدم بانتظام</li>
            <li>اتبع نظام غذائي صحي قليل السكريات</li>
            <li>مارس الرياضة بانتظام (30 دقيقة يومياً)</li>
        """
        rec_title = "التوصيات الطبية"
        rec_title_class = "text-danger"
    else:
        recommendations = """
            <li>حافظ على نمط حياة صحي ووزن مثالي</li>
            <li>مارس الرياضة بانتظام (150 دقيقة أسبوعياً)</li>
            <li>تناول غذاء متوازن غني بالألياف</li>
            <li>قلل من السكريات والمشروبات الغازية</li>
            <li>أجرِ فحص دوري للسكر كل سنة</li>
        """
        rec_title = "نصائح للوقاية"
        rec_title_class = "text-success"

    return RESULT_HTML.format(
        header_class="result-positive" if is_diabetic else "result-negative",
        icon_class="icon-positive" if is_diabetic else "icon-negative",
        icon="⚠️" if is_diabetic else "✓",
        prediction="مصاب بالسكري" if is_diabetic else "غير مصاب بالسكري",
        text_class="text-positive" if is_diabetic else "text-negative",
        probability=f"{probability*100:.1f}%",
        risk_level=risk_level,
        risk_class=risk_class,
        progress_class=progress_class,
        recommendations=recommendations,
        rec_title=rec_title,
        rec_title_class=rec_title_class
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
