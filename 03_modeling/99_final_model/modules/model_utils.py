from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import json

import os
import joblib

import numpy as np
import pandas as pd

def preprocess_and_split(X, y, test_size=0.2):
    # 스케일링 적용 O
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("✔️ Standard Scaler 적용 완료.")
    # 라벨인코딩 적용 O
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    print("✔️ LabelEncoding 적용 완료.")
    # 테스트 데이터 분리 (test_size=기본0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )
    print(f"✔️ Test Data 분리 적용 완료. size = {test_size}")
    return X_train, X_val, y_train, y_val, scaler, le_y

# ✅ XGBoost
def get_xgb_model(params=None):
    default_params = {
        'tree_method': 'hist',
        'device': 'cuda', 
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    model = XGBClassifier(**default_params)
    print("🧠 XGBoost 모델 정의 완료")
    return model

# ✅ LightGBM
def get_lgbm_model(params=None):
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
    print("💡 LightGBM 모델 정의 완료")
    return LGBMClassifier(**default_params)

# ✅ RandomForest
def get_rf_model(params=None):
    default_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1
    }
    if params:
        default_params.update(params)
    print("🌲 RandomForest 모델 정의 완료")
    return RandomForestClassifier(**default_params)

# ✅ LogisticRegression (메타모델용 등)
def get_lr_model(params=None):
    default_params = {
        "max_iter": 1000,
        "random_state": 42
    }
    if params:
        default_params.update(params)
    print("📈 LogisticRegression 모델 정의 완료")
    return LogisticRegression(**default_params)

# 모델 매핑
MODEL_FUNC_MAP = {
    "xgb": get_xgb_model,
    "rf": get_rf_model,
    "lgbm": get_lgbm_model,
    "lr": get_lr_model
}

def get_model(name: str, params: dict = None):
    name = name.lower()
    if name not in MODEL_FUNC_MAP:
        raise ValueError(f"❌ 지원하지 않는 모델 이름입니다: {name}")
    
    model_func = MODEL_FUNC_MAP[name]         # ⬅️ 함수 자체 가져옴
    model = model_func(params)                # ⬅️ 호출
    print(f"✅ 모델 생성 완료: {name.upper()}")
    return model

# 성능평가지표 함수
def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name=None):
    model_name = model_name or type(model).__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"✅ [{model_name}]")
    print("📊 성능 평가:")
    print(classification_report(y_val, y_pred))

    print("F1 Macro :", f1_score(y_val, y_pred, average='macro'))
    print("F1 Micro :", f1_score(y_val, y_pred, average='micro'))
    print("F1 Weighted :", f1_score(y_val, y_pred, average='weighted'))
    
    return model

#SMOTE 적용
def apply_smote(X_train, y_train, random_state=42, **smote_kwargs):
    """
    SMOTE를 적용해 클래스 불균형을 보정합니다. 추가 하이퍼파라미터도 지원합니다.
    
    Args:
        X_train (DataFrame or ndarray): 학습 입력 피처
        y_train (Series or ndarray): 학습 타겟
        random_state (int): 랜덤 시드 (기본값: 42)
        **smote_kwargs: SMOTE에 전달할 추가 파라미터 (k_neighbors 등)
        
    Returns:
        X_resampled, y_resampled: 클래스 균형이 맞춰진 학습 데이터셋
    """
    smote = SMOTE(random_state=random_state, **smote_kwargs)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("✅ SMOTE 적용 완료")
    print(f"  → Before: {len(y_train)} samples")
    print(f"  → After:  {len(y_resampled)} samples (class balanced)")
    
    return X_resampled, y_resampled

# 오버샘플링 함수
def apply_oversampling(X_scaled, y_encoded, strategy=None):
    """
    오버샘플링 함수 (추후 전략 수치 튜닝 가능)

    Args:
        X_scaled: 스케일링된 피처
        y_encoded: 인코딩된 타겟
        strategy: sampling_strategy 딕셔너리 (예: {0:1500, 1:3000})

    Returns:
        X_resampled, y_resampled
    """
    strategy = strategy or {0: 1500, 1: 3000}  # 기본값 지정
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_scaled, y_encoded)

    print("\ud83d\udd01 오버샘플링 완료:", pd.Series(y_resampled).value_counts())

    return np.array(X_resampled), np.array(y_resampled)

# Threshold 시각화
def plot_threshold_metrics(model, X_val, y_val, positive_class=1, thresholds=np.linspace(0.1, 0.9, 50)):
    """
    다양한 threshold에 대해 F1, Precision, Recall 시각화

    Args:
        model: 확률 예측이 가능한 학습된 모델
        X_val: 검증 데이터
        y_val: 정답 레이블
        positive_class: 관심 있는 positive 클래스 인덱스 (binary 분류 기준)
        thresholds: 실험할 threshold 리스트
    """
    y_proba = model.predict_proba(X_val)[:, positive_class]

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        y_pred_thresh = (y_proba > t).astype(int)
        f1_scores.append(f1_score(y_val, y_pred_thresh))
        precision_scores.append(precision_score(y_val, y_pred_thresh))
        recall_scores.append(recall_score(y_val, y_pred_thresh))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='o')
    plt.plot(thresholds, precision_scores, label='Precision', marker='s')
    plt.plot(thresholds, recall_scores, label='Recall', marker='^')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs F1 / Precision / Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 최적 threshold 찾기
def apply_best_threshold(model, X_val, y_val, positive_class=1, thresholds=np.linspace(0.1, 0.9, 50)):
    # 다중 클래스 y를 binary로 바꾸기
    if y_val.dtype != int and y_val.dtype != bool:
        y_val = (y_val == positive_class).astype(int)

    y_proba = model.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = 0.5
    y_pred_best = None

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            y_pred_best = y_pred

    print(f"\n🌟 최적 Threshold: {best_threshold:.3f} (F1 Score: {best_f1:.4f})")
    return best_threshold, best_f1, y_pred_best


# 모델 & 결과 저장
def save_model_and_results(model, model_name, result_df=None, save_path="./outputs"):
    """
    모델 및 결과 저장용 공통 함수
    """
    os.makedirs(save_path, exist_ok=True)
    model_path = f"{save_path}/{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"💾 모델 저장 완료: {model_path}")

    if result_df is not None:
        csv_path = f"{save_path}/experiment_results.csv"
        result_df.to_csv(csv_path, index=False)
        print(f"📁 결과 저장 완료: {csv_path}")

def save_experiment_summary(model_name, save_path, threshold=None, f1_score=None, extra_info=None):
    """
    threshold, f1, 기타 정보를 json으로 저장
    """
    os.makedirs(save_path, exist_ok=True)
    summary = {
        "model_name": model_name,
        "best_threshold": threshold,
        "best_f1": f1_score
    }
    if extra_info:
        summary.update(extra_info)

    path = os.path.join(save_path, f"{model_name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📄 요약 저장 완료: {path}")

def combine_segment_predictions(pred_e, pred_ab, pred_cd):
    """
    E, AB, CD 모델의 예측 결과를 결합하여 최종 Segment를 생성합니다.

    Parameters:
    - pred_e (list[int] or np.array): E 모델의 예측 결과 (4 또는 0)
    - pred_ab (list[int] or np.array): AB 모델의 예측 결과 (0 또는 1)
    - pred_cd (list[int] or np.array): CD 모델의 예측 결과 (2 또는 3)

    Returns:
    - final_preds (list[int]): 통합된 최종 Segment (0~4)
    """
    final_preds = []
    for e, ab, cd in zip(pred_e, pred_ab, pred_cd):
        if e == 4:
            final_preds.append(4)  # ✅ E 우선
        elif ab in [0, 1]:
            final_preds.append(ab)  # 그다음 AB
        elif cd in [2, 3]:
            final_preds.append(cd)  # 마지막 CD
        else:
            final_preds.append(-1)  # 예외 처리용 (optional)
    return final_preds

import numpy as np

def custom_threshold(proba, thresh_dict, default_class=2):
    """
    다중분류 모델의 확률 예측 결과에서 클래스별 threshold 기준으로 최종 예측 클래스 결정

    Args:
        proba: shape (n_samples, n_classes), predict_proba 결과
        thresh_dict: {class_index: threshold_value}
        default_class: 어떤 클래스도 기준 통과하지 못했을 경우 기본으로 예측할 클래스

    Returns:
        preds: shape (n_samples,), threshold 기반 예측 라벨
    """
    preds = []
    for row in proba:
        # threshold 넘는 클래스만 필터링
        valid_classes = [cls for cls, thresh in thresh_dict.items() if row[cls] >= thresh]
        if valid_classes:
            # 넘은 것 중 가장 확률 높은 클래스 선택
            pred_class = max(valid_classes, key=lambda cls: row[cls])
            preds.append(pred_class)
        else:
            preds.append(default_class)
    return np.array(preds)

import joblib

def save_xgb_model(model, path):
    """
    XGBClassifier 전체 객체 저장 (하이퍼파라미터 포함)
    """
    joblib.dump(model, path)
    print(f"✅ 모델 전체 저장 완료: {path}")

def load_xgb_model(path):
    """
    저장된 XGBClassifier 모델 불러오기
    """
    model = joblib.load(path)
    print(f"✅ 모델 로드 완료: {path}")
    return model
