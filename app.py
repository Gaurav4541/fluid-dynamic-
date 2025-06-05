import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Title of the app
st.title("💧 Fluid Dynamics Prediction App")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fluid dynamic_model.h5")
    return model

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("Upload Fluid Dynamics CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    try:
        # Make predictions
        prediction = model.predict(df)
        st.subheader("📊 Model Predictions")
        st.write(prediction)

        # Download predictions
        pred_df = pd.DataFrame(prediction, columns=[f"Output_{i+1}" for i in range(prediction.shape[1])])
        output = pd.concat([df, pred_df], axis=1)
        csv = output.to_csv(index=False)

        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
else:
    st.info("Please upload a CSV file to begin prediction.")
