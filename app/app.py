import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ======================
# LOAD MODEL
# ======================
#model = joblib.load("E:/bankruptcy-risk-system/model/xgb_model.pkl")  
model = joblib.load("model/xgb_model.pkl") 
#features = joblib.load("E:/bankruptcy-risk-system/model/features.pkl") 
features = joblib.load("model/features.pkl")

# ======================
# TITLE
# ======================
st.title("🏦 Company Default Risk & Financial Health Predictor")
# st.write(model)
# st.write(features)
st.markdown("Enter company risk indicators below:")

# ======================
# USER INPUT
# ======================

st.header("🔴 Financial Risk")

cash_flow_stress = st.number_input("Cash Flow Stress", 0.0, 100.0, 1.0)
liquidity_risk = st.number_input("Liquidity Risk", 0.0, 10.0, 1.0)
fsi = st.number_input("Financial Stability Index (FSI)", -50.0, 50.0, 5.0)

st.header("🔵 Business Risk and industry risk")

cost_pressure = st.number_input("Cost Pressure", 0.0, 10.0, 1.0)
operational_inefficiency = st.number_input("Operational Inefficiency", 0.0, 10.0, 1.0)
competitive_pressure = st.number_input("Competitive Pressure", 0.0, 20.0, 2.0)
profitability = st.number_input("Profitability", -20.0, 50.0, 10.0)

st.header("🟣 Management Risk")

management_risk = st.selectbox("Management Risk", [0, 1])

# ======================
# PREDICT BUTTON
# ======================
if st.button("Predict Risk"):

    # Create input dataframe
    input_data = pd.DataFrame([[
        cash_flow_stress,
        liquidity_risk,
        fsi,
        cost_pressure,
        operational_inefficiency,
        competitive_pressure,
        profitability,
        management_risk
    ]], columns=features)

    # ======================
    # PROBABILITY
    # ======================
    probability = model.predict_proba(input_data)[0][1]

    # ======================
    # CUSTOM THRESHOLD
    # ======================
    prediction = 1 if probability > 0.2 else 0

    # ======================
    # CREDIT SCORE
    # ======================
    credit_score = int((1 - probability) * 900)

    # ======================
    # CLASSIFICATION
    # ======================
    if credit_score >= 750:
        classification = "Excellent"
    elif credit_score >= 700:
        classification = "Good"
    elif credit_score >= 650:
        classification = "Fair"
    elif credit_score >= 550:
        classification = "Poor"
    else:
        classification = "Very Poor"

    # ======================
    # INSIGHTS & ACTIONS
    # ======================
    insights = []
    actions = []

    if cash_flow_stress > 2:
        insights.append("High cash flow stress (repayment pressure)")
        actions.append("Improve cash flow management by getting receivables earlier from clients, by negotiating longer credit period with vendors and by negotiating with lenders to reduce interest on borrowed money ")

    if liquidity_risk > 1:
        insights.append("Liquidity issues detected")
        actions.append("Increase short-term assets to increase turnover and thereafter liquidity. Convince lenders to provide short term loans at lesser interests for increasing turnover.")

    if profitability < 10:
        insights.append("Low profitability")
        actions.append("Increase profitability by positive cashflow, optimized operational efficiency, improved liquidity and risk free management.")

    if competitive_pressure > 4:
        insights.append("High competitive pressure")
        actions.append("Optimize pricing/cost strategy")

    if management_risk == 1:
        insights.append("Weak management indicators")
        actions.append("Strengthen corporate governance (for management risk)")

    if len(insights) == 0:
        insights.append("Financial condition is stable")
        actions.append("Maintain current strategy")

    # ======================
    # RESULT TABLE
    # ======================
    result_df = pd.DataFrame({
        "Metric": ["Credit Score", "Risk Classification", "Default Probability"],
        "Value": [credit_score, classification, round(probability, 2)]
    })

    st.subheader("📊 Risk Summary")
    st.table(result_df)

    # ======================
    # INSIGHT TABLE
    # ======================
    insight_df = pd.DataFrame({
        "Insights": insights,
        "Recommended Actions": actions
    })

    st.subheader("🧠 Insights & Recommendations")
    st.table(insight_df)
    # # ======================
    # # FINANCIAL HEALTH
    # # ======================
    # health_score = (1 - probability) * 100

    # if health_score > 75:
    #     status = "🟢 Healthy"
    # elif health_score > 50:
    #     status = "🟡 Moderate Risk"
    # else:
    #     status = "🔴 High Risk"

    # st.subheader("📊 Prediction Result")
    # st.write(f"Default Risk: **{prediction}**")
    # st.write(f"Probability: **{probability:.2f}**")
    # st.write(f"Financial Health Score: **{health_score:.2f}**")
    # st.write(f"Status: {status}")

    # # ======================
    # # SHAP EXPLANATION
    # # ======================
    st.subheader("🔍 Model Explanation (SHAP)")

    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # # ======================
    # # INSIGHTS
    # # ======================
    # st.subheader("🧠 Insights")

    # if cash_flow_stress > 1:
    #     st.write("- High cash flow stress impacting risk")

    # if liquidity_risk > 1:
    #     st.write("- Liquidity issues detected")

    # if profitability < 10:
    #     st.write("- Low profitability affecting financial health")

    # if management_risk == 1:
    #     st.write("- Management risk contributing to default risk")

    # # ======================
    # # ACTIONS
    # # ======================
    # st.subheader("🎯 Recommended Actions")

    # if cash_flow_stress > 1:
    #     st.write("- Improve cash flow management by getting recievables earlier,by negotiating longer credit period with vendors and By negotiating with lenders to reduce interest on borrowed money")

    # if liquidity_risk > 1:
    #     st.write("- Increase short-term assets")

    # if profitability < 10:
    #     st.write("- Improve operational efficiency and margins,by increasing productivity")

    # if management_risk == 1:
    #     st.write("- Strengthen management and governance practices")