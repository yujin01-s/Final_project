# modules/model_AB.py

from modules.feature_selector import ab_feat
from modules.model_utils import get_xgb_model, get_model, train_and_evaluate, apply_smote, apply_best_threshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import xgboost as xgb

def run_AB_modeling(df, used_cols=None, save_path = None):
    print("🔍 AB 세그먼트 모델링 시작")

    # 1. A/B 세그먼트만 필터링
    df_ab = df[df["Segment"].isin(["A", "B", 0, 1])]
    print("📌 A/B 샘플 수:", len(df_ab))
    if len(df_ab) == 0:
        raise ValueError("A/B 세그먼트 데이터가 없습니다. AB 모델 학습 불가.")

    # 2. X, y 분리
    X = df_ab[ab_feat]
    y = df_ab["Segment"]

    # 3. 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. train/val 분리
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    print("✅ 데이터 분할 완료:", X_train.shape, X_val.shape)

    # 5. 오버샘플링 (선택적으로 SMOTE 사용)
    X_train, y_train = apply_smote(X_train, y_train)

    # 6. 모델 학습 및 평가
    model = get_xgb_model()
    model.fit(X_train, y_train)
    train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="XGB_AB")

    # 7. Threshold 튜닝
    best_thresh, best_f1, y_pred = apply_best_threshold(model, X_val, y_val)

    # 8. 예측 라벨 다시 디코딩
    y_pred = le.inverse_transform(y_pred)
    y_val = le.inverse_transform(y_val)

    # 9. Booster로 저장 (JSON)
    import os
    # 저장 경로의 상위 디렉토리 확보
    save_dir = os.path.dirname(os.path.abspath(save_path))
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise NotADirectoryError(f"❗ '{save_dir}' 경로가 디렉토리가 아니라 파일입니다. 삭제하거나 경로명을 변경하세요.")
    os.makedirs(save_dir, exist_ok=True)

    booster = model.get_booster()
    booster.save_model(save_path)



    return {
        "df": df_ab,
        "features": ab_feat,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_pred": y_pred,
        "model": model,
        "threshold": best_thresh,
        "f1_score": best_f1,
    }
