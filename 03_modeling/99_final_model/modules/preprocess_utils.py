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

import pandas as pd
import numpy as np

def get_high_correlation_pairs(df, threshold=0.8):
    """
    상관계수 절댓값이 threshold 이상인 변수쌍을 표로 정렬하여 반환합니다.
    (자기 자신 제외, 중복 제거)
    """
    # 수치형만 선택
    corr = df.select_dtypes(include=['int64', 'float64']).corr()

    # 상관계수 행렬을 long-format으로 변환
    corr_pairs = corr.unstack().reset_index()
    corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']

    # 자기 자신은 제외
    corr_pairs = corr_pairs[corr_pairs['Feature_1'] != corr_pairs['Feature_2']]

    # 중복 제거 (예: A-B와 B-A 중 하나만)
    corr_pairs['sorted'] = corr_pairs.apply(lambda row: tuple(sorted([row['Feature_1'], row['Feature_2']])), axis=1)
    corr_pairs = corr_pairs.drop_duplicates(subset='sorted').drop(columns='sorted')

    # 절댓값 기준 정렬 및 필터링
    corr_pairs['AbsCorr'] = corr_pairs['Correlation'].abs()
    high_corr = corr_pairs[corr_pairs['AbsCorr'] >= threshold].sort_values(by='AbsCorr', ascending=False)

    return high_corr[['Feature_1', 'Feature_2', 'Correlation']]

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import pandas as pd

def calculate_vif(df, exclude_cols=['ID', 'Segment']):
    # 수치형 변수만 선택
    X = df.select_dtypes(include=['int64', 'float64']).drop(columns=exclude_cols, errors='ignore')

    # 결측치 제거 또는 임시 채우기 (VIF 계산은 결측 허용 안됨)
    X = X.fillna(0)

    # 정규화 (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # VIF 계산
    vif_df = pd.DataFrame()
    vif_df["feature"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X.shape[1])]

    # 정렬
    return vif_df.sort_values(by="VIF", ascending=False)

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fast_vif(X, verbose=True):
    X = X.copy()
    X = X.fillna(0)  # 결측치 처리
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    vif_dict = {}
    for i in range(X.shape[1]):
        y = X.iloc[:, i]
        X_not_i = X.drop(X.columns[i], axis=1)
        model = LinearRegression().fit(X_not_i, y)
        r2 = model.score(X_not_i, y)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_dict[X.columns[i]] = vif
        if verbose:
            print(f"{X.columns[i]}: VIF={vif:.2f}")

    vif_df = pd.DataFrame(vif_dict.items(), columns=["feature", "VIF"]).sort_values(by="VIF", ascending=False)
    return vif_df

from modules.feature_selector import safe_div

def auto_reduce_vif_features_fast(df, high_corr_table, vif_threshold=10.0, corr_threshold=0.9, top_k=20, verbose=True):
    """
    VIF 높은 컬럼 기반 고상관 변수쌍 → 차이/비율 파생변수 생성 (Top-K)
    + 상위 30개 VIF 출력 포함
    """
    df = df.copy()

    # 1. 고상관 피처쌍 먼저 필터링
    filtered_corr = high_corr_table[high_corr_table["Correlation"].abs() > corr_threshold]

    # 2. 해당 컬럼만으로 VIF 계산
    candidate_cols = set(filtered_corr["Feature_1"]) | set(filtered_corr["Feature_2"])
    candidate_cols = [col for col in candidate_cols if col in df.columns]
    vif_df = fast_vif(df[candidate_cols], verbose=False)

    # 3. 상위 30개 VIF 출력
    top30_vif = vif_df.sort_values(by="VIF", ascending=False).head(30)
    print("\n📊 [VIF 상위 30개]")
    print(top30_vif.to_string(index=False))

    # 4. VIF 높은 변수 필터
    high_vif_cols = set(vif_df[vif_df["VIF"] > vif_threshold]["feature"])

    # 5. 고상관 + VIF 높은 쌍만 Top-K
    high_corr_filtered = filtered_corr[
        (filtered_corr["Feature_1"].isin(high_vif_cols)) |
        (filtered_corr["Feature_2"].isin(high_vif_cols))
    ].sort_values(by="Correlation", ascending=False).head(top_k)

    # 6. 파생변수 생성
    created_features = []
    for _, row in high_corr_filtered.iterrows():
        f1, f2 = row["Feature_1"], row["Feature_2"]
        if f1 in df.columns and f2 in df.columns:
            diff_col = f"{f1}_minus_{f2}"
            ratio_col = f"{f1}_div_{f2}"

            df[diff_col] = df[f1] - df[f2]
            df[ratio_col] = safe_div(df[f1], df[f2] + 1e-5)
            created_features.extend([diff_col, ratio_col])

            if verbose:
                print(f"✅ Created: {diff_col}, {ratio_col}")

    return df, created_features, top30_vif

def remove_inf_div_features_fast(df, only_div_cols=True, verbose=True, threshold=0.01):
    """
    무한값(inf) 또는 NaN 비중이 높은 컬럼 제거
    - only_div_cols=True: '_div_'가 포함된 컬럼만 검사
    """
    df = df.copy()
    cols_to_remove = []

    target_cols = df.columns
    if only_div_cols:
        target_cols = [col for col in df.columns if '_div_' in col]

    for col in target_cols:
        if df[col].dtype not in ['float64', 'int64']:
            continue

        inf_ratio = np.isinf(df[col]).mean()
        nan_ratio = df[col].isna().mean()

        if inf_ratio > threshold or nan_ratio > threshold:
            cols_to_remove.append(col)
            if verbose:
                print(f"🚫 제거됨: {col} (inf_ratio={inf_ratio:.4f}, nan_ratio={nan_ratio:.4f})")

    df.drop(columns=cols_to_remove, inplace=True)
    return df, cols_to_remove

def auto_clean_high_vif_features(df, verbose=True):
    """
    VIF 상위 피처 자동 정리:
    - 신용/신판/일시불: 하나만 남김
    - 한도소진율 계열: 상위 2개 제외하고 차이/비율 파생
    - 카드론 B0M~B2M: 변화량/비율 파생 후 일부 제거
    - 방문일수/횟수: 파생 생성 후 제거
    - 포인트 건별: 차이 파생 후 제거
    """
    df = df.copy()
    removed_cols = []
    created_cols = []

    # 1. 신용/신판/일시불 중 '신용'만 유지
    candidates = ['이용건수_신판_R12M', '이용건수_일시불_R12M']
    for col in candidates:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            removed_cols.append(col)
            if verbose: print(f"🚫 제거: {col}")

    # 2. 한도소진율 계열
    hando_cols = [col for col in df.columns if '한도소진율' in col]
    hando_sorted = df[hando_cols].std().sort_values(ascending=False).index[:2].tolist()  # 상위 2개 유지
    for col in hando_cols:
        if col not in hando_sorted:
            # 파생: 차이 + 비율
            df[f"{hando_sorted[0]}_minus_{col}"] = df[hando_sorted[0]] - df[col]
            df[f"{hando_sorted[0]}_div_{col}"] = safe_div(df[hando_sorted[0]], df[col] + 1e-5)
            created_cols += [f"{hando_sorted[0]}_minus_{col}", f"{hando_sorted[0]}_div_{col}"]
            df.drop(columns=col, inplace=True)
            removed_cols.append(col)
            if verbose: print(f"🚫 제거 + 파생 생성: {col}")

    # 3. 카드론 시계열: B0M, B1M, B2M → B0M 유지
    card_cols = ['잔액_카드론_B0M', '잔액_카드론_B1M', '잔액_카드론_B2M']
    if all(col in df.columns for col in card_cols):
        df['카드론_변화량'] = df['잔액_카드론_B0M'] - df['잔액_카드론_B2M']
        df['카드론_비율'] = safe_div(df['잔액_카드론_B0M'], df['잔액_카드론_B2M'] + 1e-5)
        created_cols += ['카드론_변화량', '카드론_비율']
        for col in ['잔액_카드론_B1M', '잔액_카드론_B2M']:
            df.drop(columns=col, inplace=True)
            removed_cols.append(col)
            if verbose: print(f"🚫 제거: {col} (카드론 시계열)")

    # 4. 방문횟수/일수 앱
    if '방문횟수_앱_B0M' in df.columns and '방문일수_앱_B0M' in df.columns:
        df['방문빈도_앱'] = safe_div(df['방문횟수_앱_B0M'], df['방문일수_앱_B0M'] + 1)
        created_cols.append('방문빈도_앱')
        df.drop(columns=['방문일수_앱_B0M'], inplace=True)
        removed_cols.append('방문일수_앱_B0M')
        if verbose: print("✅ 생성: 방문빈도_앱, 제거: 방문일수_앱_B0M")

    # 5. 포인트 건별
    if '포인트_마일리지_건별_R3M' in df.columns and '포인트_마일리지_건별_B0M' in df.columns:
        df['포인트건별_변화량'] = df['포인트_마일리지_건별_R3M'] - df['포인트_마일리지_건별_B0M']
        created_cols.append('포인트건별_변화량')
        df.drop(columns=['포인트_마일리지_건별_B0M'], inplace=True)
        removed_cols.append('포인트_마일리지_건별_B0M')
        if verbose: print("✅ 생성: 포인트건별_변화량, 제거: 포인트_마일리지_건별_B0M")

    return df, removed_cols, created_cols

def remove_high_vif_features(df, vif_threshold=60.0, verbose=True):
    """
    VIF 계산 후 threshold보다 높은 컬럼 자동 제거
    반환값: 정제된 DataFrame, 제거된 컬럼 리스트, 최종 VIF DataFrame
    """
    df = df.copy()
    removed_cols = []

    while True:
        vif_df = fast_vif(df, verbose=False)
        max_vif = vif_df["VIF"].max()

        if max_vif <= vif_threshold:
            break  # 모두 threshold 이하면 종료

        # 제거할 컬럼 (VIF 가장 높은 변수 1개)
        remove_col = vif_df.iloc[0]["feature"]
        df.drop(columns=remove_col, inplace=True)
        removed_cols.append(remove_col)

        if verbose:
            print(f"🚫 제거: {remove_col} (VIF={max_vif:.2f})")

    # 최종 VIF 결과 리턴
    final_vif_df = fast_vif(df, verbose=False)
    return df, removed_cols, final_vif_df

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def fast_vif_cleaner(X, vif_threshold=100.0, remove_high_vif=True, verbose=True):
    """
    ✅ fast_vif_cleaner: 빠른 VIF 계산 및 고다중공선성 변수 제거 함수 (모듈용)
    
    Parameters:
    - X (pd.DataFrame): 수치형 변수만 포함된 입력 데이터프레임
    - vif_threshold (float): VIF 임계값 (default=100.0)
    - remove_high_vif (bool): VIF 초과 변수 반복 제거 여부
    - verbose (bool): 로그 출력 여부
    
    Returns:
    - X_cleaned (pd.DataFrame): 정제된 변수 데이터프레임
    - removed_cols (list): 제거된 컬럼명 리스트
    - final_vif (pd.DataFrame): 최종 VIF 값 DataFrame
    """

    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(axis=1, inplace=True)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    removed_cols = []

    def compute_vif(df):
        vif_dict = {}
        for i in range(df.shape[1]):
            y = df.iloc[:, i]
            X_not_i = df.drop(df.columns[i], axis=1)
            model = LinearRegression().fit(X_not_i, y)
            r2 = model.score(X_not_i, y)
            vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
            vif_dict[df.columns[i]] = vif
            if verbose:
                print(f"📌 {df.columns[i]}: VIF={vif:.2f}")
        return pd.DataFrame(vif_dict.items(), columns=["feature", "VIF"]).sort_values(by="VIF", ascending=False)

    while True:
        vif_df = compute_vif(X_scaled)
        max_vif = vif_df["VIF"].max()

        if remove_high_vif and max_vif > vif_threshold:
            to_remove = vif_df.iloc[0]["feature"]
            if verbose:
                print(f"🚫 '{to_remove}' 제거 (VIF={max_vif:.2f})")
            X_scaled.drop(columns=[to_remove], inplace=True)
            removed_cols.append(to_remove)
        else:
            break

    final_vif = compute_vif(X_scaled)

    return X_scaled, removed_cols, final_vif

def get_clean_numeric_columns(df, columns):
    """
    숫자형이고 1차원인 컬럼만 필터링
    """
    clean_cols = []
    for col in columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].ndim == 1:
                clean_cols.append(col)
            else:
                print(f"❌ 제외됨: '{col}' (ndim={df[col].ndim}, dtype={df[col].dtypes})")
        except Exception as e:
            print(f"⚠️ 오류 발생: {col}, error = {e}")
    return clean_cols