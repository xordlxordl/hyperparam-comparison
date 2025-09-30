import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import numpy as np

# RMSE 계산 함수
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 샘플 데이터 생성 (실제 데이터로 교체 필요)
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost를 위한 목적 함수 정의
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 200),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_state': 42,
        'verbose': False
    }
    model = CatBoostRegressor(**params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    return rmse(y_val, y_pred)

# Optuna 스터디 생성 및 최적화 실행
# minimize(최소 값) 모델 평가 방법에 따라 변경 maximizer(최대 값)
study = optuna.create_study(direction='minimize')
# 시도 횟수 n_trials = 50번
study.optimize(objective, n_trials=50)

print(f"모델 이름: 최적의 값 = {study.best_value}, 최적의 파라미터 = {study.best_params}")
# 최적의 파라미터와 RMSE 출력
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
