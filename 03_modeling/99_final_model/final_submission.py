# generate_submission.py

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from modules.data_loader import load_and_process
from modules.feature_selector import stage_feature_map, selected_cols, generate_cd_derived_features
from modules.model_utils import load_xgb_model
import matplotlib
import matplotlib.pyplot as plt

# === 한글 폰트 설정 ===
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# === 공통 유틸 함수 ===
def prepare_input(df, used_cols):
    for col in used_cols:
        if col not in df.columns:
            df[col] = 0
    return df[used_cols]

# === 경로 설정 ===
test_path = "../../data/통합_test_데이터.parquet"

# === Segment E 예측 ===
print("🚀 Segment E 예측 시작...")
e_model = load_xgb_model("./models/model_e.pkl")
e_threshold = 0.508

used_cols_e = e_model.feature_names_in_.tolist()
test_df, _ = load_and_process(file_path=test_path, stage="e")
test_df["ID"] = test_df["ID"].astype(str)
X_e = prepare_input(test_df, used_cols_e).drop(columns="ID", errors="ignore")
proba_e = e_model.predict_proba(X_e)[:, 1]
y_pred_e = (proba_e > e_threshold).astype(int)
test_df["Segment"] = "ABCD"
test_df.loc[y_pred_e == 1, "Segment"] = "E"
e_result = test_df[test_df["Segment"] == "E"]["ID"].copy().to_frame()
e_result["Segment"] = "E"
print("✅ Segment E 예측 완료 (E로 분류된 고객 수:", len(e_result), ")")

plt.hist(proba_e, bins=50, color='salmon')
plt.axvline(e_threshold, color='red', linestyle='--', label=f'Threshold = {e_threshold}')
plt.title("Segment E 예측 확률 분포")
plt.xlabel("E 확률")
plt.ylabel("고객 수")
plt.legend()
plt.tight_layout()
plt.savefig("e_probability_hist.png")
plt.close()
print("📊 Segment E 확률 분포 시각화 저장 완료 → e_probability_hist.png")

# === Segment ABCD 예측 ===
print("🚀 Segment ABCD 예측 시작...")
model_abcd = xgb.XGBClassifier()
model_abcd.load_model("./models/model_abcd2.json")
used_cols_abcd = joblib.load(open("./models/model_abcd_used_cols2.pkl", "rb"))

stage_feature_map["abcd"] = selected_cols
ad_test, _ = load_and_process(file_path=test_path, stage="abcd")
ad_test["ID"] = ad_test["ID"].astype(str)
ad_test = ad_test[~ad_test["ID"].isin(e_result["ID"])]
X_abcd = prepare_input(ad_test, used_cols_abcd)

d_abcd = xgb.DMatrix(X_abcd)
probs_abcd = model_abcd.get_booster().predict(d_abcd)

label_order = ["A", "B", "CD"]
th_ab = 0.15
preds_abcd_label, probs_a, probs_b, probs_cd = [], [], [], []
for prob in probs_abcd:
    probs_a.append(prob[0])
    probs_b.append(prob[1])
    probs_cd.append(prob[2])

    top_idx = prob.argmax()
    if top_idx == 0 and prob[0] >= 0.15:
        preds_abcd_label.append("A")
    elif top_idx == 1 and prob[1] >= 0.18:
        preds_abcd_label.append("B")
    else:
        preds_abcd_label.append("CD")


# A 확률 분포 시각화
plt.hist(probs_a, bins=50, color='skyblue')
plt.axvline(th_ab, color='red', linestyle='--', label=f'Threshold = {th_ab}')
plt.title("Segment A 예측 확률 분포")
plt.xlabel("A 확률")
plt.ylabel("고객 수")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("a_probability_hist.png")
plt.close()
print("📊 Segment A 확률 분포 시각화 저장 완료 → a_probability_hist.png")

# B 확률 분포 시각화
plt.hist(probs_b, bins=50, color='lightblue')
plt.axvline(th_ab, color='red', linestyle='--', label=f'Threshold = {th_ab}')
plt.title("Segment B 예측 확률 분포")
plt.xlabel("B 확률")
plt.ylabel("고객 수")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("b_probability_hist.png")
plt.close()
print("📊 Segment B 확률 분포 시각화 저장 완료 → b_probability_hist.png")

# CD 확률 분포 시각화
plt.hist(probs_cd, bins=50, color='lightgray')
plt.title("Segment CD 예측 확률 분포")
plt.xlabel("CD 확률")
plt.ylabel("고객 수")
plt.grid()
plt.tight_layout()
plt.savefig("cd_softmax_probability_hist.png")
plt.close()
print("📊 Segment CD softmax 확률 분포 시각화 저장 완료 → cd_softmax_probability_hist.png")

# === Segment 결과 저장 ===
abcd_result = ad_test[["ID"]].copy()
abcd_result["Segment_before_CD"] = preds_abcd_label
abcd_result["Segment"] = preds_abcd_label.copy()

# === Segment CD 예측 ===
print("🚀 Segment CD 재예측 시작...")
cd_ids = abcd_result[abcd_result["Segment_before_CD"] == "CD"]
cd_test = ad_test[ad_test["ID"].isin(cd_ids["ID"])]
cd_test = generate_cd_derived_features(cd_test)

model_cd = xgb.XGBClassifier()
model_cd.load_model("./models/model_cd.json")
used_cols_cd = joblib.load(open("./models/model_cd_used_cols.pkl", "rb"))
X_cd = prepare_input(cd_test, used_cols_cd)
d_cd = xgb.DMatrix(X_cd)
probs_cd_model = model_cd.get_booster().predict(d_cd)
preds_cd = (probs_cd_model > 0.5).astype(int)

le_cd = LabelEncoder()
le_cd.fit(["C", "D"])
preds_cd_label = le_cd.inverse_transform(preds_cd)

cd_result = cd_test[["ID"]].copy()
cd_result["Segment"] = preds_cd_label
print("✅ Segment CD 보정 완료")

# CD 모델 확률 분포 시각화 (C 확률)
plt.hist(probs_cd_model, bins=50, color='lightgreen')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.title("Segment CD 모델 확률 분포 (C 확률 기준)")
plt.xlabel("C 확률")
plt.ylabel("고객 수")
plt.legend()
plt.tight_layout()
plt.savefig("cd_model_probability_hist.png")
plt.close()
print("📊 Segment CD 모델 확률 분포 시각화 저장 완료 → cd_model_probability_hist.png")

# === 병합 및 우선순위 기반 최종 결정 ===
ab_preds = abcd_result[abcd_result["Segment_before_CD"].isin(["A", "B"])]
cd_only = abcd_result[abcd_result["Segment_before_CD"] == "CD"]
cd_only = cd_only[~cd_only["ID"].isin(ab_preds["ID"])]

submission_all = pd.concat([
    e_result[["ID", "Segment"]],
    ab_preds[["ID", "Segment"]],
    cd_result[["ID", "Segment"]]
], axis=0)

def resolve_segment(group):
    if "E" in group["Segment"].values:
        return "E"
    counts = group["Segment"].value_counts()
    if "A" in counts or "B" in counts:
        return "A" if counts.get("A", 0) >= counts.get("B", 0) else "B"
    if "C" in counts and "D" in counts:
        return "C" if counts["C"] >= counts["D"] else "D"
    for fallback in ["CD", "C", "D"]:
        if fallback in counts:
            return fallback
    return counts.idxmax()

submission = submission_all.groupby("ID").apply(resolve_segment).reset_index()
submission.columns = ["ID", "Segment"]

final_ids = test_df[["ID"]].drop_duplicates().copy()
submission = final_ids.merge(submission, on="ID", how="left")
submission["Segment"] = submission["Segment"].fillna("E")

assert submission["ID"].duplicated().sum() == 0, "❌ 여전히 중복된 ID가 존재합니다!"
submission = submission.sort_values("ID").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print("✅ 제출 파일이 저장되었습니다 → submission.csv")
print("📊 최종 Segment 분포:")
print(submission["Segment"].value_counts())

# === 전체 세그먼트 분류 프로세스 도식 저장 ===
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")
plt.title("Segment 분류 전체 프로세스 요약", fontsize=16, fontweight='bold')

flow = [
    ("🔹 test_df", "Segment E 모델 → 확률 > 0.508 → Segment E 예측"),
    ("나머지", "Segment ABCD 모델 → A/B 확률 > 0.15 → A/B/그 외 CD 예측"),
    ("CD로 분류된 고객", "Segment CD 모델 → C vs D 재예측"),
    ("모든 결과 병합", "Segment 우선순위 병합 (E > A/B > C/D)"),
    ("최종 결과", "submission.csv 저장")
]

y_start = 0.9
for i, (left, right) in enumerate(flow):
    ax.text(0.05, y_start - i*0.15, left, fontsize=13, va="center", fontweight="bold")
    ax.annotate(right, xy=(0.3, y_start - i*0.15), fontsize=12, va="center")

plt.tight_layout()
plt.savefig("segment_process_flow.png")
plt.close()
print("🗺️ 전체 예측 구조 도식 저장 완료 → segment_process_flow.png")
