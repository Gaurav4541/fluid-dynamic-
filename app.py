import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
@st.cache_resource
def load_trained_model():
    return load_model("fluid dynamic_model.h5")

model = load_trained_model()

st.set_page_config(page_title="Fluid Dynamics Predictor", layout="centered")

st.title("💧 Fluid Dynamics Prediction App")
st.markdown("Upload your fluid data or enter manually to get predictions!")

# Sample CSV input
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    try:
        prediction = model.predict(df)
        st.success("✅ Prediction Successful!")
        st.write("🧠 Model Output:")
        st.dataframe(prediction)
    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")

st.markdown("---")
st.subheader("Or Enter Input Manually")

# Manual input fields
dudy = st.number_input("dudy", value=0.0)
dvdx = st.number_input("dvdx", value=0.0)
dvdy = st.number_input("dvdy", value=0.0)
dudt = st.number_input("dudt", value=0.0)
dvdt = st.number_input("dvdt", value=0.0)

input_data = np.array([[dudy, dvdx, dvdy, dudt, dvdt]])

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
