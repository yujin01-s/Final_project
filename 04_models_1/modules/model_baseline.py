# modules/model_cd.py

from modules.feature_selector import baseline_feat
from modules.model_utils import get_xgb_model, get_model, train_and_evaluate, apply_smote, apply_best_threshold, save_model_and_results, apply_oversampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def run_baseline_modeling(df, used_cols, save_path=None):
    print("🔍 전체 세그먼트 모델링 시작")

    # 1. 파생변수 생성
    #df = generate_base_derived_features(df) 

    # 2. X, y 분리
    X = df[baseline_feat]
    y = df["Segment"]

    # 3. 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. train/val 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    print("✅ 데이터 분할 완료:", X_train.shape, X_val.shape)

    # 5. 오버샘플링 (선택적으로 SMOTE 사용)
    X_train, y_train = apply_oversampling(X_train, y_train)

    # 6. 모델 학습 및 평가
    model = get_xgb_model()  # 또는 get_model("xgb", {...})
    model.fit(X_train, y_train)
    train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="XGB_base")

    # 7. 예측 라벨 다시 디코딩
    y_pred = le.inverse_transform(y_pred)
    y_val = le.inverse_transform(y_val)

    # 8. 저장
    if save_path:
        save_model_and_results(model, model_name="XGB_base", save_path=save_path)

    return {
        "df": df,
        "features": baseline_feat,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_pred": y_pred,
        "model": model,
        "f1_score": best_f1
    }
