import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit UI
st.title("Student Score Prediction App")

# Sidebar Input Fields
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.sidebar.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
test_preparation_course = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Prediction Button
if st.sidebar.button("Predict Score"):
    # Create data instance
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert input to DataFrame
    pred_df = data.get_data_as_data_frame()
    st.write("### Input Data:")
    st.write(pred_df)

    # Load Model and Predict
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(pred_df)

    # Display Prediction
    st.success(f"Predicted Math Score: {prediction[0]:.2f}")
