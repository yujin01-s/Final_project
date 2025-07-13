import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from modules.feature_selector import stage_feature_map

from modules.preprocess_utils import (
    map_categorical_columns,
    encode_categorical_columns
)

def load_and_process(file_path: str,
                     stage: str = None,
                     selected_cols: list = None,
                     base_cols: list = ["ID", "Segment"]):
    """
    파일을 불러오고, stage 또는 selected_cols 기반으로 컬럼을 추출 + 범주형 인코딩

    Parameters:
    - file_path (str): .parquet 또는 .csv
    - stage (str): stage_feature_map에서 키를 찾아 컬럼 자동 매핑
    - selected_cols (list): 직접 지정한 컬럼 리스트 (stage가 None일 때 사용)
    - base_cols (list): 항상 포함할 기준 컬럼

    Returns:
    - df_final (pd.DataFrame): base_cols + 가공된 컬럼
    - used_columns (list): 가공 대상 컬럼 리스트 (base 제외)
    """

    # 파일 불러오기
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("지원되지 않는 파일 형식입니다.")
    print("데이터 불러오기 완료.")

    # ✅ stage 기반 selected_cols 설정
    if stage is not None:
        if stage not in stage_feature_map:
            raise ValueError(f"'{stage}'는 존재하지 않습니다.")
        selected_cols = stage_feature_map[stage]

    # ✅ base + selected_cols 기준 필터링
    if selected_cols is not None:
        keep_cols = list(set(base_cols + selected_cols))
        df = df[keep_cols]

    # ✅ 중복 컬럼 제거
    df = df.loc[:, ~df.columns.duplicated()]

    # ✅ 사용자 정의 매핑
    df = map_categorical_columns(df)

    # ✅ 남은 범주형 인코딩
    object_cols = df.select_dtypes(include='object').columns.tolist()
    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        #print(f"✅ Label Encoding 적용: {col}")
    print("✅ 범주형 인코딩 완료")
    # 🍕 사용된 컬럼 정리
    used_columns = [col for col in df.columns if col not in base_cols]
    df.head(5)
    return df, used_columns


def test_load(path: str, scaler: StandardScaler = None):
    """
    테스트 데이터 로딩 + 전처리 (타겟 없음)
    Args:
        path (str): 통합된 parquet 테스트 데이터 경로
        scaler (StandardScaler): train에서 사용한 스케일러 객체

    Returns:
        X_test (DataFrame): 전처리된 테스트 데이터
    """
    print(f"📂 테스트 데이터 로딩 중: {path}")
    df = pd.read_parquet(path)

    X = df.copy()
    X = X.loc[:, ~X.columns.duplicated()]

    # 전처리 동일하게 적용
    X = map_categorical_columns(X)
    X = encode_categorical_columns(X)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    if scaler:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)
        print("📏 테스트 데이터 스케일링 적용 완료")

    print(f"✅ test_load 완료: X={X.shape}")
    return X
