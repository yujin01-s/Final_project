import pandas as pd
import glob
import os

# 1. 시트별 경로 정의
table_paths = {
    "1.회원정보": "data/train/1.회원정보",
    "8.성과정보": "data/train/8.성과정보"
}

# 2. 데이터 불러오기
merged_data = {}

for name, path in table_paths.items():
    files = glob.glob(os.path.join(path, "*.parquet"))
    if len(files) > 0:
        df = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
        merged_data[name] = df
        print(f"{name} ✅ 불러오기 완료: {df.shape}")
    else:
        print(f"{name} ⚠️ 파일 없음")

# 3. 필요한 데이터 추출 및 병합
member_df = merged_data["1.회원정보"]
result_df = merged_data["8.성과정보"]

# ID, Segment만 추출
segment_df = member_df[['ID', 'Segment']].drop_duplicates(subset='ID')

# ID 기준 병합
result_df_with_segment = pd.merge(result_df, segment_df, on='ID', how='left')

# 4. 병합 결과 확인
print(f"병합 후 행 수: {len(result_df_with_segment)}")
print(f"Segment 컬럼 결측치 수: {result_df_with_segment['Segment'].isnull().sum()}")

# 5. CSV 파일로 저장
save_path = "data/성과정보_with_segment.csv"
result_df_with_segment.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"📁 CSV 저장 완료: {save_path}")