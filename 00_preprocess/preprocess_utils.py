import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# segment_target 예시: ["C", "D"], ["A", "B"], ["E", "A", "B", "C", "D"]
def prepare_data(df, segment_target):
    X_all = df[df['Segment'].isin(segment_target)].copy()
    y_all = LabelEncoder().fit_transform(X_all['Segment'])
    return train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
    """
    # 1단계: E vs 나머지
    X_train, X_val, y_train, y_val = prepare_data(df, ['E', 'A', 'B', 'C', 'D'])

    # 2단계: AB vs CD
    X_train, X_val, y_train, y_val = prepare_data(df, ['A', 'B', 'C', 'D'])

    # 3단계: A vs B
    X_train, X_val, y_train, y_val = prepare_data(df, ['A', 'B'])

    # 4단계: C vs D
    X_train, X_val, y_train, y_val = prepare_data(df, ['C', 'D'])
    """

# 데이터 준비 (파생변수 생성)
def data_derive(df, stage='cd'):
    df = df.copy()

    # 파생변수 자동 생성만 수행
    if stage == 'cd':
        df = generate_cd_derived_features(df)
    # elif stage == 'ab':
    #     df = generate_ab_derived_features(df)
    # elif stage == 'e':
    #     df = generate_e_derived_features(df)

    return df

# VIF 기반 피처 제거
def remove_high_vif(df, features, fixed=[], threshold=10.0):
    features = features.copy()
    while True:
        X = df[features].fillna(0)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        max_vif = vif.drop(labels=fixed).max()
        if max_vif > threshold:
            to_remove = vif.drop(labels=fixed).idxmax()
            features.remove(to_remove)
        else:
            break
    return features

# 예측 확률 생성 함수
def predict_proba_on_fixed_val(X_train, y_train, X_val, features):
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train[features], y_train)
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train_res, y_train_res)
    return model.predict_proba(X_val[features])[:, 1]

# 파생변수 생성
def safe_div(a, b):
    return np.where(b == 0, 0, a / b)

# 범주화 전처리
def map_categorical_columns(df, verbose=True):
    """
    미리 정의된 매핑 기준에 따라 범주형 컬럼들을 수치형으로 변환합니다.
    처리 컬럼: 거주시도명, 연회비발생카드수_B0M, 한도증액횟수_R12M, 이용금액대,
              할인건수_R3M, 할인건수_B0M, 방문횟수_PC_R6M, 방문횟수_앱_R6M, 방문일수_PC_R6M
    """

    # 1. 거주시도명 → 수도권 여부
    capital_area = ['서울', '경기', '인천']
    if '거주시도명' in df.columns:
        df['거주시도명'] = df['거주시도명'].apply(lambda x: 1 if x in capital_area else 0)

    # 2. 연회비발생카드수_B0M
    mapping = {"0개": 0, "1개이상": 1}
    if '연회비발생카드수_B0M' in df.columns:
        df['연회비발생카드수_B0M'] = df['연회비발생카드수_B0M'].map(mapping).astype(int)
        if verbose: print("[연회비발생카드수_B0M] 인코딩 완료")

    # 3. 한도증액횟수_R12M
    mapping = {"0회": 0, "1회이상": 1}
    if '한도증액횟수_R12M' in df.columns:
        df['한도증액횟수_R12M'] = df['한도증액횟수_R12M'].map(mapping).astype(int)
        if verbose: print("[한도증액횟수_R12M] 인코딩 완료")

    # 4. 이용금액대 (중간값 기준: 만원 단위)
    mapping = {
        "09.미사용": 0,
        "05.10만원-": 5,
        "04.10만원+": 20,
        "03.30만원+": 40,
        "02.50만원+": 75,
        "01.100만원+": 150
    }
    if '이용금액대' in df.columns:
        df['이용금액대'] = df['이용금액대'].map(mapping)
        if verbose: print("[이용금액대] 중간값 인코딩 완료")

   # 5. 할인건수 인코딩
    discount_map = {
        "1회 이상": 1,
        "10회 이상": 10,
        "20회 이상": 20,
        "30회 이상": 30,
        "40회 이상": 40
    }
    for col in ['할인건수_R3M', '할인건수_B0M']:
        if col in df.columns:
            df[col] = df[col].map(discount_map).astype(int)
            if verbose: print(f"[{col}] 인코딩 완료")

    # 6. 방문횟수 및 방문일수 인코딩
    visit_map = {
        "1회 이상": 1,
        "10회 이상": 10,
        "20회 이상": 20,
        "30회 이상": 30,
        "40회 이상": 40,
        "50회 이상": 50,
        "60회 이상": 60,
        "70회 이상": 70,
        "80회 이상": 80
    }

    visit_cols = ['방문횟수_PC_R6M', '방문횟수_앱_R6M', '방문일수_PC_R6M']
    for col in visit_cols:
        if col in df.columns:
            df[col] = df[col].map(visit_map).astype(int)
            if verbose: print(f"[{col}] 인코딩 완료")

    return df

# CD 파생변수 생성
def generate_cd_derived_features(df):
    df = df.copy()
    df['오프라인_소비급등비율'] = safe_div(df['이용금액_오프라인_B0M'], df['이용금액_오프라인_R6M'] / 6)
    df['일시불_최근성지표'] = safe_div(df['이용금액_일시불_B0M'], df['최대이용금액_일시불_R12M'] + 1)
    df['소비_평균대최대비율'] = safe_div((df['이용금액_R3M_신용'] + 1) / 3, df['최대이용금액_일시불_R12M'] + 1)
    df['청구_최근성지표'] = safe_div(df['정상청구원금_B0M'], df['정상청구원금_B5M'] + 1)
    df['소진율_차이'] = df['잔액_신판평균한도소진율_r6m'] - df['잔액_신판ca평균한도소진율_r6m']
    df['카드집중도'] = safe_div(df['_1순위카드이용금액'], df['이용금액_R3M_신용'] + df['이용금액_R3M_신용체크'] + 1)
    df['입금청구비율'] = safe_div(df['정상입금원금_B5M'], df['정상청구원금_B5M'] + 1)
    df['일시불_금액비율'] = safe_div(df['이용금액_일시불_B0M'], df['평잔_일시불_6M'] + 1)
    df['할인전이자율_평균'] = (df['RV일시불이자율_할인전'] + df['CA이자율_할인전']) / 2
    return df