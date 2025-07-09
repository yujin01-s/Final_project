# 선별된 피처 리스트 정리

import numpy as np
import pandas as pd

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

# CD 분류모델용 피처
cd_final_features = [
    '정상청구원금_B5M', '이용금액_R3M_신용체크', '정상입금원금_B5M', 
    '최대이용금액_일시불_R12M', '이용가맹점수', '이용금액_오프라인_R6M', 
    '평잔_일시불_6M', '잔액_신판ca평균한도소진율_r6m', 'RV일시불이자율_할인전', '오프라인_소비급등비율', '일시불_최근성지표', 
    '소비_평균대최대비율', '청구_최근성지표', '소진율_차이', '카드집중도', '입금청구비율', '일시불_금액비율'
    ]

stage_feature_map = {
    "cd": cd_final_features
    # ,
    # "ab": ab_final_features,
    # "e": e_final_features
}

"""
from feature_selector import stage_feature_map

# 'cd' 단계의 피처만 불러오기
selected_features = stage_feature_map['cd']

"""