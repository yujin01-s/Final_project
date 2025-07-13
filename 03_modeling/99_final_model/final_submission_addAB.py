# generate_submission.py (A/B/CD 분류 후 A/B 보정 적용, 오류 수정 + 각 모델별 피처 일치)

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from modules.data_loader import load_and_process
from modules.feature_selector import stage_feature_map, selected_cols, generate_cd_derived_features
from modules.model_utils import load_xgb_model
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def prepare_input(df, used_cols):
    for col in used_cols:
        if col not in df.columns:
            df[col] = 0
    return df[used_cols]

test_path = "../../data/통합_test_데이터.parquet"

print("\n🚀 Segment E 예측 시작...")
e_model = load_xgb_model("./models/model_e.pkl")
e_threshold = 0.508
used_cols_e = e_model.feature_names_in_.tolist()
test_df, _ = load_and_process(file_path=test_path, stage="e")
test_df["ID"] = test_df["ID"].astype(str)
if "base_ym" in test_df.columns:
    test_df["ID"] = test_df["ID"] + "_" + test_df["base_ym"].astype(str)
X_e = prepare_input(test_df, used_cols_e).drop(columns="ID", errors="ignore")
proba_e = e_model.predict_proba(X_e)[:, 1]
y_pred_e = (proba_e > e_threshold).astype(int)
test_df["Segment"] = "ABCD"
test_df.loc[y_pred_e == 1, "Segment"] = "E"
e_result = test_df[test_df["Segment"] == "E"]["ID"].copy().to_frame()
e_result["Segment"] = "E"
print("✅ Segment E 예측 완료 (E로 분류된 고객 수:", len(e_result), ")")

print("\n🚀 Segment ABCD 예측 시작...")
booster_abcd = xgb.Booster()
booster_abcd.load_model("./models/model_abcd2.json")
used_cols_abcd = joblib.load(open("./models/model_abcd_used_cols2.pkl", "rb"))

stage_feature_map["abcd"] = selected_cols
abcd_test, _ = load_and_process(file_path=test_path, stage="abcd")
abcd_test["ID"] = abcd_test["ID"].astype(str)
if "base_ym" in abcd_test.columns:
    abcd_test["ID"] = abcd_test["ID"] + "_" + abcd_test["base_ym"].astype(str)
abcd_test = abcd_test[~abcd_test["ID"].isin(e_result["ID"])]
X_abcd = prepare_input(abcd_test, used_cols_abcd)
d_abcd = xgb.DMatrix(X_abcd)
probs_abcd = booster_abcd.predict(d_abcd)

label_order = ["A", "B", "C", "D"]
th_a = 0.15
th_b = 0.18
preds_abcd_label = []
for prob in probs_abcd:
    if len(prob) < 4:
        preds_abcd_label.append("CD")
        continue
    top_idx = prob.argmax()
    if top_idx >= len(label_order):
        preds_abcd_label.append("CD")
        continue
    top_label = label_order[top_idx]
    if top_label == "A" and prob[0] >= th_a:
        preds_abcd_label.append("A")
    elif top_label == "B" and prob[1] >= th_b:
        preds_abcd_label.append("B")
    elif top_label in ["C", "D"]:
        preds_abcd_label.append("CD")
    else:
        preds_abcd_label.append("CD")

abcd_test = abcd_test.copy()
# ✅ 길이 체크 후 적용
if len(preds_abcd_label) != len(abcd_test):
    raise ValueError(f"예측 결과 길이 불일치: preds_abcd_label({len(preds_abcd_label)}), abcd_test({len(abcd_test)})")
abcd_test = abcd_test.copy()
abcd_test["Segment_before_AB"] = preds_abcd_label

print("\n🔍 Segment A/B 보정 시작...")
ab_model = xgb.XGBClassifier()
ab_model.load_model("./models/model_ab.json")
used_cols_ab = joblib.load(open("./models/model_ab_used_cols.pkl", "rb"))
ab_test = abcd_test.copy()
ab_prob_threshold = 1e-6
ab_candidate_mask = (probs_abcd[:, 0] >= ab_prob_threshold) | (probs_abcd[:, 1] >= ab_prob_threshold)
ab_test = ab_test.loc[ab_candidate_mask].copy()
X_ab = prepare_input(ab_test, used_cols_ab).drop(columns="ID", errors="ignore")
X_ab = prepare_input(ab_test, used_cols_ab).drop(columns="ID", errors="ignore")
if len(X_ab) == 0:
    ab_result = pd.DataFrame(columns=["ID", "Segment"])
else:
    proba_ab = ab_model.predict_proba(X_ab)[:, 1]
    ab_pred = (proba_ab > 1e-6).astype(int)
    le_ab = LabelEncoder()
    le_ab.fit(["A", "B"])
    ab_pred_label = le_ab.inverse_transform(ab_pred)
    ab_result = ab_test[["ID"]].copy()
    ab_result["Segment"] = ab_pred_label

print("\n🔍 Segment CD 보정 시작...")
cd_ids = abcd_test[abcd_test["Segment_before_AB"] == "CD"]["ID"].tolist()
cd_test, _ = load_and_process(file_path=test_path, stage="cd")
cd_test["ID"] = cd_test["ID"].astype(str)
if "base_ym" in cd_test.columns:
    cd_test["ID"] = cd_test["ID"] + "_" + cd_test["base_ym"].astype(str)
cd_test = cd_test[cd_test["ID"].isin(cd_ids)]
cd_test = generate_cd_derived_features(cd_test)
model_cd = xgb.XGBClassifier()
model_cd.load_model("./models/model_cd.json")
used_cols_cd = joblib.load(open("./models/model_cd_used_cols.pkl", "rb"))
X_cd = prepare_input(cd_test, used_cols_cd)
probs_cd_model = model_cd.predict_proba(X_cd)[:, 0]
preds_cd = (probs_cd_model > 0.5).astype(int)
le_cd = LabelEncoder()
le_cd.fit(["C", "D"])
preds_cd_label = le_cd.inverse_transform(preds_cd)
cd_result = cd_test[["ID"]].copy()
cd_result["Segment"] = preds_cd_label

for df_ in [e_result, ab_result, cd_result]:
    df_["ID"] = df_["ID"].astype(str)

submission_all = pd.concat([
    e_result[["ID", "Segment"]],
    ab_result[["ID", "Segment"]],
    cd_result[["ID", "Segment"]]
], axis=0).drop_duplicates(subset="ID", keep="last")

submission = test_df[["ID"]].drop_duplicates().copy()
submission = submission.merge(submission_all, on="ID", how="left")
submission["Segment"] = submission["Segment"].fillna("E")
submission = submission.sort_values("ID").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)
print("\n✅ 제출 파일 저장 완료 → submission.csv")
print("📊 최종 Segment 분포:")
print(submission["Segment"].value_counts())
