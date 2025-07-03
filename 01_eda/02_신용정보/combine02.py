import pandas as pd
import glob
import os

# 시트 번호 및 정확한 폴더명 매핑 (띄어쓰기 제거 + 경로 반영)
table_paths = {
    "1.회원정보": "data/train/1.회원정보",
    "2.신용정보": "data/train/2.신용정보",
    "3.승인매출정보": "data/train/3.승인매출정보",
    "4.청구입금정보": "data/train/4.청구입금정보",
    "5.잔액정보": "data/train/5.잔액정보",
    "6.채널정보": "data/train/6.채널정보",
    "7.마케팅정보": "data/train/7.마케팅정보",
    "8.성과정보": "data/train/8.성과정보"
}

# 결과 저장용 딕셔너리
merged_data = {}

# 각 테이블 통합
for name, path in table_paths.items():
    files = glob.glob(os.path.join(path, "*.parquet"))
    if len(files) > 0:
        df = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
        merged_data[name] = df
        print(f"{name} ✅ 불러오기 완료: {df.shape}")
    else:
        print(f"{name} ⚠️ 파일 없음")

# 예시: 성과정보 테이블 미리 보기
merged_data["8.성과정보"].head()

# 신용정보 시트를 새로운 변수에 넣는다.
credit_df = merged_data["2.신용정보"]

# 도수분포 확인
credit_df.info()

# 1. 필요한 컬럼만 추출 (ID, Segment)
segment_df = member_df[['ID', 'Segment']].drop_duplicates(subset='ID')

# 2. ID 기준으로 credit_df에 병합
credit_df_with_segment = pd.merge(credit_df, segment_df, on='ID', how='left')

# 3. 결과 확인
print(f"병합 후 행 수: {len(credit_df_with_segment)}")
print(f"Segment 컬럼 결측치 수: {credit_df_with_segment['Segment'].isnull().sum()}")