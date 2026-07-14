import streamlit as st
import pandas as pd
import joblib

from xgboost import XGBClassifier


@st.cache_resource
def load_model():

    preprocessor = joblib.load(
        r"D:\projects\churn_preprocessor.pkl"
    )

    model = XGBClassifier()

    model.load_model(
        r"D:\projects\churn_xgboost_model.json"
    )

    return preprocessor, model


preprocessor, model = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Customer Churn Intelligence")

st.markdown(
    """
    Predict customer churn risk using an XGBoost machine learning model
    trained on over **440,000 customer records**.
    
    Enter the customer's behavioral, subscription, and payment details
    to generate a real-time churn risk assessment.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.markdown(
        """
        **Algorithm:** XGBoost Classifier
        
        **Training Records:** 440K+
        
        **Features:** 10
        
        **Task:** Binary Classification
        
        **Target:** Customer Churn
        """
    )

    st.divider()

    st.caption(
        "Built using Python, Pandas, Scikit-learn, "
        "XGBoost and Streamlit."
    )


# --------------------------------------------------
# CUSTOMER INPUT SECTION
# --------------------------------------------------

st.header("Customer Profile")

col1, col2 = st.columns(2)


# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------

with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=70,
        value=35
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    tenure = st.slider(
        "Tenure (months)",
        min_value=1,
        max_value=60,
        value=24
    )

    usage_frequency = st.slider(
        "Usage Frequency",
        min_value=1,
        max_value=30,
        value=15,
        help="Number of times the customer uses the service."
    )

    support_calls = st.slider(
        "Support Calls",
        min_value=0,
        max_value=10,
        value=2
    )


# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------

with col2:

    payment_delay = st.slider(
        "Payment Delay (days)",
        min_value=0,
        max_value=30,
        value=5
    )

    subscription_type = st.selectbox(
        "Subscription Type",
        [
            "Basic",
            "Standard",
            "Premium"
        ]
    )

    contract_length = st.selectbox(
        "Contract Length",
        [
            "Monthly",
            "Quarterly",
            "Annual"
        ]
    )

    total_spend = st.number_input(
        "Total Spend",
        min_value=0.0,
        max_value=1000.0,
        value=500.0,
        step=10.0
    )

    last_interaction = st.slider(
        "Days Since Last Interaction",
        min_value=1,
        max_value=30,
        value=10
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Analyze Churn Risk",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_type],
        "Contract Length": [contract_length],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })
    # Transform raw customer input using the fitted preprocessor
    processed_input = preprocessor.transform(input_data)

    # Make prediction using the XGBoost model
    prediction = model.predict(processed_input)[0]

    # Get probability of customer churn
    churn_probability = model.predict_proba(
        processed_input
    )[0][1]

    st.header("Risk Assessment")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Model Churn Score",
            f"{churn_probability:.2%}"
        )

    with metric2:

        if churn_probability < 0.30:
            risk_level = "Low"

        elif churn_probability < 0.70:
            risk_level = "Medium"

        else:
            risk_level = "High"

        st.metric(
            "Risk Level",
            risk_level
        )

    with metric3:

        prediction_label = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction_label
        )


    # ----------------------------------------------
    # RISK PROGRESS BAR
    # ----------------------------------------------

    st.subheader("Churn Risk Score")

    st.progress(float(churn_probability))


    # ----------------------------------------------
    # RISK MESSAGE
    # ----------------------------------------------

    if risk_level == "High":

        st.error(
            "🔴 High churn risk detected. "
            "Immediate customer retention intervention "
            "is recommended."
        )

    elif risk_level == "Medium":

        st.warning(
            "🟡 Moderate churn risk detected. "
            "The customer should be monitored and "
            "considered for targeted engagement."
        )

    else:

        st.success(
            "🟢 Low churn risk. "
            "The customer currently shows strong "
            "retention indicators."
        )


    # ----------------------------------------------
    # KEY RISK INDICATORS
    # ----------------------------------------------

    st.subheader("Key Risk Indicators")

    risk_factors = []

    if support_calls >= 5:
        risk_factors.append(
            "High number of customer support calls"
        )

    if payment_delay >= 15:
        risk_factors.append(
            "Frequent or prolonged payment delays"
        )

    if contract_length == "Monthly":
        risk_factors.append(
            "Monthly contract provides lower customer commitment"
        )

    if last_interaction >= 20:
        risk_factors.append(
            "Long period since the customer's last interaction"
        )

    if total_spend < 300:
        risk_factors.append(
            "Relatively low total customer spending"
        )

    if risk_factors:

        for factor in risk_factors:
            st.write(f"⚠️ {factor}")

    else:

        st.write(
            "No major rule-based risk indicators detected."
        )


    # ----------------------------------------------
    # BUSINESS RECOMMENDATION
    # ----------------------------------------------

    st.subheader("Recommended Action")

    if risk_level == "High":

        st.info(
            """
            **Recommended retention strategy:**
            
            - Contact the customer proactively.
            - Investigate unresolved support issues.
            - Offer a personalized retention incentive.
            - Consider encouraging migration to a longer-term contract.
            """
        )

    elif risk_level == "Medium":

        st.info(
            """
            **Recommended engagement strategy:**
            
            - Monitor future support interactions.
            - Send personalized engagement offers.
            - Address potential payment or service concerns early.
            """
        )

    else:

        st.info(
            """
            **Recommended strategy:**
            
            - Maintain regular customer engagement.
            - Continue providing personalized service.
            - Consider loyalty rewards for long-term retention.
            """
        )