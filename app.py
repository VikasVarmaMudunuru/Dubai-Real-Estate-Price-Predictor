import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Page Config ---
st.set_page_config(
    page_title="Dubai Real Estate Price Predictor",
    page_icon="🏙️",
    layout="centered"
)

# --- Load Model & Supporting Files ---
@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.pkl")

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

model        = load_model()
feature_cols = load_json("feature_cols.json")
enc_maps     = load_json("encoding_maps.json")
area_list    = load_json("area_list.json")
prop_types   = load_json("property_types.json")

# --- Header ---
st.title("🏙️ Dubai Real Estate Price Predictor")
st.markdown(
    "Predict property sale prices in Dubai using a machine learning model "
    "trained on **1,153,228 official transactions** from the Dubai Land Department (2004–2025)."
)
st.divider()

# --- Input Form ---
st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("📍 Area / Neighbourhood", options=area_list)
    property_type = st.selectbox("🏠 Property Type", options=prop_types)
    procedure_area = st.number_input("📐 Property Size (sqm)", min_value=10, max_value=10000, value=100, step=10)

with col2:
    rooms = st.selectbox("🛏️ Rooms", options=[
        "Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R",
        "5 B/R", "6 B/R", "7 B/R+", "Unknown"
    ])
    has_parking = st.radio("🚗 Parking Included?", options=["Yes", "No"], horizontal=True)
    is_offplan = st.radio("🏗️ Property Status", options=["Ready", "Off-Plan"], horizontal=True)

col3, col4 = st.columns(2)
with col3:
    year = st.selectbox("📅 Transaction Year", options=list(range(2025, 2019, -1)))
with col4:
    month = st.selectbox("🗓️ Transaction Month", options=list(range(1, 13)),
                          format_func=lambda x: [
                              "Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

st.divider()

# --- Predict Button ---
if st.button("🔮 Predict Price", use_container_width=True, type="primary"):

    # Build input dict
    quarter = (month - 1) // 3 + 1
    parking_val  = 1 if has_parking == "Yes" else 0
    offplan_val  = 1 if is_offplan == "Off-Plan" else 0

    input_dict = {
        "procedure_area": procedure_area,
        "has_parking":    parking_val,
        "is_offplan":     offplan_val,
        "year":           year,
        "month":          month,
        "quarter":        quarter,
    }

    # Apply target encoding
    global_mean = 2_121_227  # fallback mean from training data

    cat_inputs = {
        "area_name_en":          area,
        "property_type_en":      property_type,
        "property_sub_type_en":  "Unknown",
        "property_usage_en":     "Unknown",
        "rooms_en":              rooms,
        "nearest_metro_en":      "Unknown",
    }

    for col, val in cat_inputs.items():
        enc_col = f"{col}_encoded"
        if enc_col in feature_cols:
            mapping = enc_maps.get(col, {})
            input_dict[enc_col] = mapping.get(val, global_mean)

    # Build feature vector in correct order
    input_vector = pd.DataFrame([[input_dict.get(col, 0) for col in feature_cols]],
                                  columns=feature_cols)

    # Predict (model was trained on log price)
    log_pred = model.predict(input_vector)[0]
    price_aed = np.expm1(log_pred)

    # Confidence range ± 15%
    low  = price_aed * 0.85
    high = price_aed * 1.15

    # --- Results ---
    st.success("Prediction Complete!")

    res1, res2, res3 = st.columns(3)
    with res1:
        st.metric("💰 Estimated Price", f"AED {price_aed:,.0f}")
    with res2:
        st.metric("📉 Low Estimate", f"AED {low:,.0f}")
    with res3:
        st.metric("📈 High Estimate", f"AED {high:,.0f}")

    price_m = price_aed / 1e6
    st.info(
        f"**Summary:** A {procedure_area} sqm "
        f"{'off-plan' if offplan_val else 'ready'} {property_type.lower()} "
        f"in {area} is estimated at **AED {price_m:.2f}M** "
        f"(AED {price_aed/procedure_area:,.0f}/sqm)"
    )

    st.caption(
        "⚠️ This is an ML estimate based on historical DLD transactions. "
        "Actual prices may vary based on floor, view, finishing, and market conditions."
    )

# --- Footer ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 13px;'>
        Model: XGBoost &nbsp;|&nbsp; R² = 0.7992 &nbsp;|&nbsp;
        Data: Dubai Land Department (1.15M transactions) &nbsp;|&nbsp;
        Built by <a href='https://www.linkedin.com/in/vikas-varma-mudunuru' target='_blank'>Vikas Varma Mudunuru</a>
    </div>
    """,
    unsafe_allow_html=True
)
