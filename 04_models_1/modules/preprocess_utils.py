import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE 
from xgboost import XGBClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_process(file_path: str,
                     stage: str = None,
                     selected_cols: list = None,
                     base_cols: list = ["ID", "Segment"],
                     stage_feature_map: dict = None):
    """
    파일을 불러오고, 선택된 컬럼 + base 컬럼만 유지한 뒤 범주형 인코딩 처리.

    1차: map_categorical_columns → 매핑
    2차: 남은 범주형에 LabelEncoding 적용

    Returns:
    - df_final (pd.DataFrame): 전처리 완료된 데이터프레임
    - used_columns (list): 사용된 가공 대상 컬럼 리스트
    """
    if stage is not None:
        if stage_feature_map is None:
            raise ValueError("stage_feature_map이 함수에 전달되지 않았습니다.")
        if stage not in stage_feature_map:
            raise ValueError(f"'{stage}'는 stage_feature_map에 존재하지 않습니다.")
        selected_cols = stage_feature_map[stage]
        
    # 1. 파일 불러오기
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다. .parquet 또는 .csv")

    # 2. 선택된 컬럼 + base_cols 기준 필터링
    if selected_cols is not None:
        keep_cols = list(set(base_cols + selected_cols))
        df = df[keep_cols]

    # 3. 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]

    # 4. 사용자 정의 매핑 함수 적용
    df = map_categorical_columns(df)

    # 5. 여전히 남은 범주형 컬럼(Label Encoding)
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"✅ Label Encoding 적용: {col}")

    # 6. 사용된 가공 대상 컬럼 기록 (base 제외)
    used_columns = [col for col in df.columns if col not in base_cols]

    return df, used_columns

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

# 라벨인코딩
def encode_categorical_columns(df):
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"🔵 {col} 인코딩 완료")
    return df

# X, y 분리 및 범주형인코딩 + 결측치 처리 (mean)
def seperateX_y(df, selected_feature):
    """
    피처 및 타겟 분리 + 범주형 인코딩 + 결측치 처리
    Args:
        df: 전처리 완료된 DataFrame
        selected_feature: 사용할 피처 리스트

    Returns:
        X: 전처리 완료된 피처
        y: 타겟
    """
    # 1. 분리
    X = df[selected_feature].copy()
    y = df["Segment"]

    # 2. 범주형 인코딩
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 3. 결측치 처리
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y


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