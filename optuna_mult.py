import optuna
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# RMSE 계산 함수
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 샘플 데이터 생성 (실제 데이터로 교체 필요)
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna를 위한 목적 함수 정의
def objective(trial):
    model_name = trial.suggest_categorical('model', ['XGBoost', 'LightGBM', 'CatBoost'])
    if model_name == 'XGBoost':
        params = {
            'subsample': trial.suggest_float('subsample', 0.7, 0.8),
            'n_estimators': trial.suggest_int('n_estimators', 100, 150),
            'min_child_weight': trial.suggest_int('min_child_weight', 6, 8),
            'max_depth': trial.suggest_int('max_depth', 8, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.15),
            'gamma': trial.suggest_float('gamma', 0.1, 0.2),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.5),
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
    elif model_name == 'LightGBM':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 40),
            'n_estimators': trial.suggest_int('n_estimators', 100, 200),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8),
            'random_state': 42,
            'force_row_wise': True,
            'verbose': -1
        }
        model = LGBMRegressor(**params)
    elif model_name.startswith('CatBoost'):
        params = {
            'iterations': trial.suggest_int('iterations', 150, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
            'depth': trial.suggest_int('depth', 6, 8),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 8, 9),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        model = CatBoostRegressor(**params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return rmse(y_val, y_pred)
    
# Optuna 스터디 생성 및 최적화 실행
# minimize(최소 값) 모델 평가 방법에 따라 변경 maximizer(최대 값)
study = optuna.create_study(direction='minimize')
# 시도 횟수 n_trials = 50번 
study.optimize(objective, n_trials=50)

print(f"모델 이름: 최적의 값 = {study.best_value}, 최적의 파라미터 = {study.best_params}")
# 최적의 파라미터와 그때의 RMSE 출력
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
