from modules.feature_selector import selected_cols
from modules.model_utils import get_xgb_model, get_model, train_and_evaluate, apply_smote, apply_best_threshold, save_model_and_results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def run_E_modeling(df, used_cols, save_path=None):
    print("🔍 E 세그먼트 분리 모델링 시작")
    # 1. Segment 필터링 (Segment가 숫자일 경우)
    df = df[df['Segment'].isin([0, 1, 2, 3, 4])].copy()
    print("📊 [train_e] Segment 분포:\n", df["Segment"].value_counts())

    # 2. 이진 타겟 생성 (Segment 4 = 'E')
    X = df[selected_cols]
    y = df['Segment'].map(lambda x: 1 if x == 4 else 0)
    print("📊 [train_e] y 분포 (1=E, 0=Others):\n", y.value_counts())
    y_encoded = y

    # 3. train/val 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    print("✅ 데이터 분할 완료:", X_train.shape, X_val.shape)


    # 5. 오버샘플링 (선택적으로 SMOTE 사용)
    print("y_train value counts:", pd.Series(y_train).value_counts())
    X_train, y_train = apply_smote(X_train, y_train)

    # 6. 모델 학습 및 평가
    model = get_xgb_model()  # 또는 get_model("xgb", {...})
    model.fit(X_train, y_train)
    train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name="XGB_E")

    # E 모델링 단계에서
    y_val_bin = y_val
    best_thresh, best_f1, y_pred = apply_best_threshold(model, X_val, y_val_bin)

    # 8. 저장
    if save_path:
        save_model_and_results(model, model_name="XGB_E", save_path=save_path)

    return {
        "df": df,
        "features": selected_cols,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_pred": y_pred,
        "model": model,
        "threshold": best_thresh,
        "f1_score": best_f1,
    }
