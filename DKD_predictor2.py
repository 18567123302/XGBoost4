import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("xgb_model.pkl")

feature_names = ["Hb", "PLT", "ALT", "BUN", "UA", "HDL"]

st.title("Risk Prediction of Diabetic Nephropathy in Elderly Patients with Type 2 Diabetes in the Community")
st.title("社区老年二型糖尿病患者糖尿病肾病风险预测")

hb  = st.number_input("Hb (Hemoglobin) (血红蛋白) <g/L>:", min_value=50, max_value=200, value=120)
Plt = st.number_input("PLT (Platelets) (血小板) <10^9/L>:", min_value=10, max_value=500, value=280)
alt = st.number_input("ALT (Alanine Aminotransferase) (血清谷丙转氨酶) <U/L>:", value=25)
bun = st.number_input("BUN (Blood urea nitrogen) (血尿素氮) <mmol/L>:", min_value=0, max_value=50, value=5)
ua  = st.number_input("UA (Uric Acid) (尿酸) <μmol/L>:", min_value=100, max_value=800, value=350)
hdl = st.number_input("HDL (高密度脂蛋白胆固醇) <mmol/L>:", value=2.0, step=0.1)

if st.button("Predict (预测)"):
    feature_values = [hb, Plt, alt, bun, ua, hdl]
    X = pd.DataFrame([feature_values], columns=feature_names)
    features = X.values.astype(float)

    predicted_class = int(model.predict(features)[0])
    predicted_proba = model.predict_proba(features)[0]

    st.write(f"**Predicted Class (预测类别):** {predicted_class}")
    st.write(f"**Prediction Probabilities (预测概率):** {predicted_proba}")

    probability = float(predicted_proba[predicted_class]) * 100

    if predicted_class == 1:
        advice = (
            "According to this model, you may have a higher risk of developing diabetic nephropathy.\n"
            f"The predicted probability of developing diabetic nephropathy is {probability:.1f}%.\n\n"
            "根据模型预测，您可能存在较高的糖尿病肾病发病风险。\n"
            f"模型预测的糖尿病肾病发病概率为 {probability:.1f}%。\n"
            "建议您尽快就医，以进行更详细的诊断和采取适当的治疗措施。"
        )
    else:
        advice = (
            "According to the model, your risk of diabetic nephropathy is low.\n"
            f"The predicted probability of not having diabetic nephropathy is {probability:.1f}%.\n\n"
            "根据模型预测，您的糖尿病肾病风险较低。\n"
            f"模型预测的无糖尿病肾病概率为 {probability:.1f}%。\n"
            "建议您继续保持健康的生活方式，并定期观察健康状况。如有任何异常症状，请及时就医。"
        )

    st.markdown(advice)

    # ---- SHAP (放在按钮内部，避免刷新就跑) ----
    try:
        explainer = shap.TreeExplainer(model.get_booster())
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):  # 二分类常见
            shap_vec = shap_values[1][0]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            shap_vec = shap_values[0]
            base_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        plt.figure()
        shap.force_plot(base_value, shap_vec, X.iloc[0, :], matplotlib=True)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}")
