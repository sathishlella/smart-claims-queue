from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def get_baseline_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=1337, max_iter=1000))
    ])

def get_strong_model():
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=1337,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        print("XGBoost not found, falling back to sklearn GradientBoostingClassifier")
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=1337
        )

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
