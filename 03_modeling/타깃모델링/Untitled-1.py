# %%
import pandas as pd

# 파일 경로
file_path = "../../data/통합_train_데이터.parquet"
df = pd.read_parquet(file_path)

# %%
top_ab = ['할부금액_3M_R12M',
 '이용금액_할부_무이자_R12M',
 '이용건수_할부_무이자_R12M',
 '정상입금원금_B0M',
 '이용금액_오프라인_R6M',
 '정상청구원금_B0M',
 '이용금액_오프라인_R3M',
 '_1순위카드이용금액',
 '평잔_일시불_해외_6M',
 '승인거절건수_입력오류_R3M',
 '청구금액_R3M',
 '이용금액_일시불_R12M',
 '이용금액_할부_무이자_R3M',
 '이용금액_할부_무이자_R6M',
 '정상입금원금_B5M',
 '이용금액_할부_R12M',
 '마일_적립포인트_R3M',
 '정상청구원금_B2M',
 '포인트_마일리지_환산_B0M',
 '청구금액_B0',
 '정상입금원금_B2M',
 '할부건수_무이자_3M_R12M',
 '정상청구원금_B5M',
 '청구금액_R6M',
 '여유_숙박이용금액',
 '최대이용금액_일시불_R12M',
 '_1순위업종_이용금액',
 '잔액_할부_B0M',
 '할부금액_무이자_3M_R12M',
 '잔액_할부_무이자_B0M']

# %%
top_cde = ['이용금액_일시불_R3M',
 '이용금액_R3M_신용체크',
 '정상입금원금_B0M',
 '이용금액_오프라인_R6M',
 '정상청구원금_B0M',
 '이용건수_신용_R6M',
 '이용금액_오프라인_R3M',
 '이용건수_일시불_R12M',
 '_1순위카드이용금액',
 '이용금액_오프라인_B0M',
 '청구금액_R3M',
 '이용금액_일시불_R12M',
 '이용금액_R3M_신용',
 '정상입금원금_B5M',
 '이용건수_오프라인_B0M',
 '정상청구원금_B2M',
 '이용건수_신판_R12M',
 '청구금액_B0',
 '정상입금원금_B2M',
 '이용금액_일시불_B0M',
 '정상청구원금_B5M',
 '청구금액_R6M',
 '최대이용금액_일시불_R12M',
 '이용가맹점수',
 '이용건수_신용_R12M',
 '이용금액_일시불_R6M']

# %%
# 📌 PC1~PC5에서 반복적으로 중요한 변수:
pca_cols = ['CA이자율_할인전', 'CL이자율_할인전', 'RV_평균잔액_R3M', 'RV일시불이자율_할인전', 'RV최소결제비율', 'RV현금서비스이자율_할인전', '방문월수_앱_R6M', '방문일수_앱_B0M', '방문일수_앱_R6M', '방문횟수_앱_B0M', '방문후경과월_앱_R6M', '이용금액_R3M_신용', '이용금액_R3M_신용체크', '이용금액_일시불_B0M', '이용금액대', '일시불ONLY전환가능여부', '잔액_리볼빙일시불이월_B0M', '잔액_일시불_B0M', '잔액_일시불_B1M', '잔액_일시불_B2M', '잔액_카드론_B0M', '잔액_카드론_B1M', '잔액_카드론_B2M', '잔액_카드론_B3M', '잔액_카드론_B4M', '잔액_카드론_B5M', '정상청구원금_B0M', '정상청구원금_B2M', '정상청구원금_B5M', '청구금액_B0', '청구금액_R3M', '청구금액_R6M', '최종카드론_대출금액', '카드론이용금액_누적', '평잔_RV일시불_3M', '평잔_RV일시불_6M', '평잔_일시불_3M', '평잔_일시불_6M', '평잔_카드론_3M', '평잔_카드론_6M', '평잔_할부_3M', '홈페이지_금융건수_R3M', '홈페이지_금융건수_R6M', '홈페이지_선결제건수_R3M', '홈페이지_선결제건수_R6M']

# %%
selected_cols=(top_ab + top_cde + pca_cols)
selected_cols = list(dict.fromkeys(selected_cols))

# %%
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

# %%
df = map_categorical_columns(df)

# %%
ab_cols=(top_ab + pca_cols)
ab_cols = list(dict.fromkeys(ab_cols))

# %%
# 📌 Segment 라벨 기준: A/B → 1, C/D/E → 0
df['is_ab'] = df['Segment'].map(lambda x: 1 if x in ['A', 'B'] else 0)

# 👇 예시용 피처 컬럼 리스트 (사용 중인 걸로 대체하세요)
X = df[selected_cols]  # 선택된 피처들
y = df['is_ab']       # 이진 타겟

# ✅ Train / Validation 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ✅ 모델 후보 정의
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix

models = {
#    "Logistic": LogisticRegression(max_iter=1000, class_weight='balanced'),
#    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(y == 0).sum() / (y == 1).sum()),
#    "LightGBM": LGBMClassifier(class_weight='balanced')
}

# ✅ 모델 학습 및 성능 평가
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds)
    print(f"\n📌 [{name}] F1-score: {f1:.4f}")
    print(classification_report(y_val, preds))
    print(confusion_matrix(y_val, preds))


# %%
# A/B 추려진 데이터에서
ab_df = df[df['is_ab'] == 1].copy()

# 타겟: A=0, B=1
y_ab = ab_df['Segment'].map({'A': 0, 'B': 1})

# 전체 피처셋
X_ab_all = ab_df[ab_cols]

# %%
from xgboost import XGBClassifier
import pandas as pd

# 간단한 XGBoost 모델 학습
xgb_ab = XGBClassifier()
xgb_ab.fit(X_ab_all, y_ab)

# 중요도 추출
importances = pd.Series(xgb_ab.feature_importances_, index=X_ab_all.columns)
top_features_ab = importances.sort_values(ascending=False).head(20).index.tolist()

print("📌 A/B에서 가장 중요한 피처 Top 20:")
print(top_features_ab)

# %%
# 선택된 피처만 사용
X_ab = X_ab_all[top_features_ab]

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train_ab, X_val_ab, y_train_ab, y_val_ab = train_test_split(X_ab, y_ab, stratify=y_ab, test_size=0.2, random_state=42)

model_ab_final = XGBClassifier()
model_ab_final.fit(X_train_ab, y_train_ab)
preds_ab = model_ab_final.predict(X_val_ab)

print(classification_report(y_val_ab, preds_ab))

# %%
# ab_df를 X_ab와 같은 인덱스 기준으로 정렬
ab_df = ab_df.loc[X_ab.index].copy()

# 예측값 저장
ab_preds = model_ab_final.predict(X_ab)
ab_df['Segment_pred'] = ab_preds
ab_df['Segment_pred'] = ab_df['Segment_pred'].map({0: 'A', 1: 'B'})


# %%
cde_cols=(top_cde + pca_cols)
cde_cols = list(dict.fromkeys(cde_cols))

# %%
cols_needed = cde_cols + ['Segment', 'ID']
cde_df = df.loc[df['is_ab'] == 0, cols_needed]


# %%
# 타겟: Segment 문자 → 숫자 라벨
y_cde = cde_df['Segment'].map({'C': 0, 'D': 1, 'E': 2})

# 입력 피처
X_cde = cde_df[cde_cols]

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# 학습용 데이터 분리
X_train_cde, X_val_cde, y_train_cde, y_val_cde = train_test_split(
    X_cde, y_cde, stratify=y_cde, test_size=0.2, random_state=42
)

# 다중 분류 모델 학습
model_cde = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
model_cde.fit(X_train_cde, y_train_cde)

# 예측 및 평가
preds_cde = model_cde.predict(X_val_cde)
print(classification_report(y_val_cde, preds_cde))

# %%
# 전체 예측 (학습용 데이터 전체로)
cde_preds = model_cde.predict(X_cde)

# 숫자 → 문자 라벨 복원
cde_df['Segment_pred'] = pd.Series(cde_preds, index=X_cde.index)
cde_df['Segment_pred'] = cde_df['Segment_pred'].map({0: 'C', 1: 'D', 2: 'E'})

# %%
# ab_df는 이전에 Segment_pred가 포함된 A/B 예측 결과
final_df = pd.concat([ab_df, cde_df])

# ID 기준 정렬
final_df = final_df.sort_values('ID')

# 제출 파일 생성
submission = final_df[['ID', 'Segment_pred']].rename(columns={'Segment_pred': 'Segment'})
submission.to_csv('final_submission.csv', index=False)

# %%
test_df = pd.read_parquet("../../data/통합_test_데이터.parquet")
X_test = test_df[selected_cols].copy()  # 또는 ab_cols, cde_cols 기반

# ID 백업
test_ids = test_df['ID']

# %%
from sklearn.preprocessing import LabelEncoder

# test_df에 범주형 컬럼 변환 적용
test_df = map_categorical_columns(test_df)
test_df.update(test_df)  # 인코딩된 컬럼만 반영

ab_proba_test = models.predict_proba(X_test)[:, 1]

# 기준값 설정 (ex: 0.5)
ab_pred_test = (ab_proba_test >= 0.5).astype(int)

# %%



