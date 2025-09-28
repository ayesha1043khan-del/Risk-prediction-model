
import os, io, pandas as pd, streamlit as st
from predict_helper import load_model, predict_df

st.set_page_config(page_title="IT Project Risk Dashboard", layout="wide")
st.title("🔎 IT Project Risk Dashboard")
st.caption("Upload model .pkl, then predict risk from form or CSV")

with st.sidebar:
    st.header("Model")
    model = None
    src = st.selectbox("Load model from", ["Upload .pkl", "Path"])
    if src=="Upload .pkl":
        up = st.file_uploader("Upload trained pipeline (.pkl)", type=["pkl"])
        if up:
            model = load_model(io.BytesIO(up.read()))
            st.success("Model loaded from upload.")
    else:
        p = st.text_input("Model path", value="models/best_model__LogisticRegression.pkl")
        if os.path.exists(p):
            model = load_model(p); st.success(f"Loaded: {p}")
        else:
            st.info("Enter a valid path or upload a .pkl.")

st.header("Predict")
mode = st.radio("Input mode", ["Manual form","Upload CSV"])

def run_predict(df):
    global model
    if model is None:
        st.error("Load model first.")
        return
    out = predict_df(model, df)
    st.dataframe(out, use_container_width=True)
    st.write("Counts:", out["risk_level_predicted"].value_counts())

if mode=="Manual form":
    c1,c2,c3 = st.columns(3)
    with c1:
        domain = st.selectbox("domain", ["Fintech","Retail","Healthcare","Telecom","Government","E-commerce"])
        methodology = st.selectbox("methodology", ["Agile","Waterfall","Hybrid"])
        team_location = st.selectbox("team_location", ["Co-located","Distributed","Hybrid"])
        vendor_contract_type = st.selectbox("vendor_contract_type", ["T&M","Fixed Bid","Managed Services"])
    with c2:
        planned_budget_lakhs = st.number_input("planned_budget_lakhs", 1.0, 5000.0, 50.0)
        planned_duration_days = st.number_input("planned_duration_days", 30, 1000, 180)
        team_size = st.number_input("team_size", 1, 200, 12)
        team_experience_years_avg = st.number_input("team_experience_years_avg", 0.0, 30.0, 5.0)
    with c3:
        tech_complexity = st.slider("tech_complexity (1-5)", 1, 5, 3)
        requirements_changes = st.number_input("requirements_changes", 0, 500, 12)
        open_bugs_count = st.number_input("open_bugs_count", 0, 500, 10)
        overdue_tasks_pct = st.number_input("overdue_tasks_pct (0..1)", 0.0, 1.0, 0.1)

    if st.button("Predict"):
        row = dict(domain=domain, methodology=methodology, team_location=team_location, vendor_contract_type=vendor_contract_type,
                   planned_budget_lakhs=planned_budget_lakhs, planned_duration_days=planned_duration_days,
                   team_size=team_size, team_experience_years_avg=team_experience_years_avg, tech_complexity=tech_complexity,
                   requirements_changes=requirements_changes, open_bugs_count=open_bugs_count, vendor_dependency=0,
                   stakeholder_count=8, offshore_ratio=0.3, sprint_length_days=14 if methodology=="Agile" else 0,
                   story_points_planned=200 if methodology=="Agile" else 0, completed_rate=0.9, issue_churn_rate=0.12,
                   reopens_rate=0.05, dependency_count=5, overdue_tasks_pct=overdue_tasks_pct, requirements_clarity=3,
                   risk_register_count=6, past_similar_success_rate=0.7, buffer_days=10, schedule_slippage_pct=overdue_tasks_pct*30,
                   cost_variance_pct=overdue_tasks_pct*20)
        df = pd.DataFrame([row])
        run_predict(df)

else:
    up = st.file_uploader("Upload CSV with training feature columns", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())
        if st.button("Predict for uploaded CSV"):
            run_predict(df)
