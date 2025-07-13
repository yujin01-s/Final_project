from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pandas as pd

import pandas as pd
from modules.model_utils import train_and_evaluate

def run_single_model_experiment(X_train, y_train, X_val, y_val, model_name, model):
    """
    단일 모델 학습 및 평가 실행
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')

    print(f"\n✅ [{model_name}]")
    print(classification_report(y_val, y_pred))

    return {"Model": model_name, "Macro_F1": round(f1, 4)}


def run_custom_experiments(X_train, y_train, X_val, y_val, model_configs):
    """
    여러 개별 모델 실험 반복 실행

    Args:
        model_configs (list): (name, model) 튜플의 리스트
    Returns:
        pd.DataFrame: 성능 비교표
    """
    results = []

    for name, model in model_configs:
        result = run_single_model_experiment(X_train, y_train, X_val, y_val, model_name=name, model=model)
        results.append(result)

    df_result = pd.DataFrame(results).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
    print("\n📊 전체 성능 비교 결과:")
    print(df_result)
    return df_result

def run_voting_ensemble(X_train, y_train, X_val, y_val, model_configs, weights=None, voting='soft'):
    """
    Weighted Voting 앙상블 실험 실행

    Args:
        model_configs (list): (name, model) 튜플의 리스트
        weights (list): 각 모델에 부여할 가중치
        voting (str): 'soft' 또는 'hard'

    Returns:
        dict: {'Model': name, 'Macro_F1': score}
    """
    model_name = "VotingEnsemble"
    ensemble_model = VotingClassifier(
        estimators=model_configs,
        voting=voting,
        weights=weights,
        n_jobs=-1
    )
    return run_single_model_experiment(X_train, y_train, X_val, y_val, model_name, ensemble_model)

def run_stacking_ensemble(X_train, y_train, X_val, y_val, base_models, meta_model_name="lr"):
    """
    Stacking 앙상블 실험 실행

    Args:
        base_models (list): (name, model) 튜플 리스트
        meta_model_name (str): 'lr', 'xgb' 등 메타모델 이름

    Returns:
        dict: 평가 결과
    """
    final_estimator = get_model(meta_model_name)
    model_name = f"Stacking_{meta_model_name.upper()}"

    stack_model = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    return run_single_model_experiment(X_train, y_train, X_val, y_val, model_name, stack_model)
