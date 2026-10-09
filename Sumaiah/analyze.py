import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================

package = joblib.load("house_price_model.pkl")

model = package["model"]
feature_columns = package["feature_columns"]
model_name = package["model_name"]


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🏠 House Price Prediction")

st.write(
    "Predict the estimated price of a house using Machine Learning."
)


# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("🏡 Enter House Details")


col1, col2 = st.columns(2)


# ==========================================
# LEFT COLUMN
# ==========================================

with col1:

    area = st.number_input(
        "📐 Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1800,
        step=50
    )

    bedrooms = st.number_input(
        "🛏️ Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "🛁 Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )


# ==========================================
# RIGHT COLUMN
# ==========================================

with col2:

    age = st.number_input(
        "📅 House Age (Years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

    parking = st.selectbox(
        "🚗 Parking",
        ["No", "Yes"]
    )

    location = st.selectbox(
        "📍 Location",
        [
            "Kochi",
            "Kozhikode",
            "Thiruvananthapuram",
            "Thrissur",
            "Kannur"
        ]
    )


# ==========================================
# PREDICT BUTTON
# ==========================================

st.write("")

if st.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    # ======================================
    # CREATE INPUT DATA
    # ======================================

    input_data = pd.DataFrame({

        "Area_sqft": [area],

        "Bedrooms": [bedrooms],

        "Bathrooms": [bathrooms],

        "Age_Years": [age],

        "Parking_Yes": [
            1 if parking == "Yes" else 0
        ],

        "Location_Kannur": [
            1 if location == "Kannur" else 0
        ],

        "Location_Kochi": [
            1 if location == "Kochi" else 0
        ],

        "Location_Kozhikode": [
            1 if location == "Kozhikode" else 0
        ],

        "Location_Thiruvananthapuram": [
            1 if location == "Thiruvananthapuram" else 0
        ],

        "Location_Thrissur": [
            1 if location == "Thrissur" else 0
        ]
    })


    # ======================================
    # MATCH MODEL COLUMNS
    # ======================================

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]

    price_lakhs = prediction / 100000


    # ======================================
    # PREDICTION RESULT
    # ======================================

    st.markdown("### 🎯 Prediction Result")

    st.success(
        f"🏠 ESTIMATED HOUSE PRICE: ₹{price_lakhs:.2f} Lakhs"
    )


    # ======================================
    # HOUSE DETAILS
    # ======================================

    st.markdown("### 📋 House Details")


    c1, c2, c3 = st.columns(3)


    with c1:

        st.write("📐 **Area**")
        st.write(f"{area} sq.ft")


    with c2:

        st.write("🛏️ **Bedrooms**")
        st.write(bedrooms)


    with c3:

        st.write("🛁 **Bathrooms**")
        st.write(bathrooms)


    c4, c5, c6 = st.columns(3)


    with c4:

        st.write("📅 **Age**")
        st.write(f"{age} years")


    with c5:

        st.write("🚗 **Parking**")
        st.write(parking)


    with c6:

        st.write("📍 **Location**")
        st.write(location)


    # ======================================
    # MODEL INFORMATION
    # ======================================

    st.info(
        f"🤖 Model Used: {model_name}"
    )


# ==========================================
# DISCLAIMER
# ==========================================

st.caption(
    "⚠️ This is an educational Machine Learning demonstration. "
    "The predicted price is only an estimate and should not be "
    "treated as a professional property valuation."
)