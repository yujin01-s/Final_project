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



# 청구입금정보정보 시트를 새로운 변수에 넣는다.
bill_df = merged_data["4.청구입금정보"]

# 도수분포 확인
bill_df.info()


# 새로운 변수에 지정해서 만들어준다.
member_df = merged_data["1.회원정보"]

# 1. 필요한 컬럼만 추출 (ID, Segment)
segment_df = member_df[['ID', 'Segment']].drop_duplicates(subset='ID')

# 2. ID 기준으로 credit_df에 병합
bill_df_with_segment = pd.merge(bill_df, segment_df, on='ID', how='left')

# 3. 결과 확인
print(f"병합 후 행 수: {len(bill_df_with_segment)}")
print(f"Segment 컬럼 결측치 수: {bill_df_with_segment['Segment'].isnull().sum()}")


# 포인트, 마일리지부터는 새로운 파일 생성
import pandas as pd
import glob
import os

# 1. 시트별 경로 정의
table_paths = {
    "1.회원정보": "data/train/1.회원정보",
    "4.청구입금정보": "data/train/4.청구입금정보"
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
bill_df = merged_data["4.청구입금정보"]

# ID, Segment만 추출
segment_df = member_df[['ID', 'Segment']].drop_duplicates(subset='ID')

# ID 기준 병합
bill_df_with_segment = pd.merge(bill_df, segment_df, on='ID', how='left')

# 4. 병합 결과 확인
print(f"병합 후 행 수: {len(bill_df_with_segment)}")
print(f"Segment 컬럼 결측치 수: {bill_df_with_segment['Segment'].isnull().sum()}")

# 5. CSV 파일로 저장
save_path = "data/청구입금정보_with_segment.csv"
bill_df_with_segment.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"📁 CSV 저장 완료: {save_path}")
