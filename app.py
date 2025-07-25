import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt # Although not directly used for the new plot, keeping it as it was in original
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Load the trained model and test data ---
# IMPORTANT: You need to ensure 'best_model.pkl' and 'test_data.pkl' exist
# 'test_data.pkl' should contain a tuple (xtest, ytest)
try:
    model = pickle.load(open("best_model.pkl", "rb"))
    # Assuming test_data.pkl contains the preprocessed xtest and ytest
    xtest, ytest = pickle.load(open("test_data.pkl", "rb"))
except FileNotFoundError:
    st.error("Error: Model or test data files not found. Please ensure 'best_model.pkl' and 'test_data.pkl' are in the same directory.")
    st.stop() # Stop the app if files are not found

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

st.title("Will You Earn More Than 50K?")
st.markdown("Enter your details and we'll predict if you're likely to earn **above or below 50K** 💵")

# Sidebar Inputs
st.sidebar.header("Modify Your Profile")

age = st.sidebar.slider("Age", 18, 75, 30)
hours = st.sidebar.slider("Hours per Week", 1, 99, 40)
educational_num = st.sidebar.slider("Education Level (1-16)", 1, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 5000, 0)

workclass = st.sidebar.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Others'])
marital = st.sidebar.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed'])
occupation = st.sidebar.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Protective-serv', 'Armed-Forces', 'Others'])
relationship = st.sidebar.selectbox("Relationship", ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
race = st.sidebar.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
gender = st.sidebar.radio("Gender", ['Male', 'Female'])
country = st.sidebar.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Others'])

# Build DataFrame for prediction
input_dict = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [100000],  # placeholder - ensure this matches your training data's fnlwgt handling
    'education-num': [educational_num], # Renamed to match your notebook's column name
    'marital-status': [marital],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours],
    'native-country': [country]
}

input_df = pd.DataFrame(input_dict)

# IMPORTANT: Preprocessing for new input_df MUST match training preprocessing
# This part assumes your 'model' object (loaded from best_model.pkl)
# handles the preprocessing internally (e.g., if it's a Pipeline)
# OR you need to load your scaler and encoders and apply them here
# For demonstration, I'm assuming the model can handle raw input or
# that you'll add the preprocessing steps before prediction.
# If your model expects numerical input, you MUST apply LabelEncoding and MinMaxScaler
# to input_df before passing it to model.predict().
# Example (conceptual, requires loading encoders and scaler):
# for col, encoder in loaded_encoders.items():
#    if col in input_df.columns:
#        input_df[col] = encoder.transform(input_df[col])
# input_df_scaled = loaded_scaler.transform(input_df)


st.markdown("### 🧾 Your Input Summary")

# Format the display keys nicely
pretty_keys = {
    'age': 'Age',
    'workclass': 'Workclass',
    'fnlwgt': 'FNLWGT',
    'education-num': 'Education Level (Num)', # Updated key
    'marital-status': 'Marital Status',
    'occupation': 'Occupation',
    'relationship': 'Relationship',
    'race': 'Race',
    'gender': 'Gender',
    'capital-gain': 'Capital Gain',
    'capital-loss': 'Capital Loss',
    'hours-per-week': 'Hours per Week',
    'native-country': 'Native Country'
}

# Sort and display in 2 columns
col1, col2 = st.columns(2)
items = list(pretty_keys.items())

for i, (key, label) in enumerate(items):
    value = input_dict[key][0]
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        st.markdown(f"**{label}:** {value}")

if st.button("PREDICT"):
    # Make sure the input_df is preprocessed correctly before prediction
    # If your model expects numerical input from LabelEncoder and MinMaxScaler,
    # you MUST apply those transformations here to input_df
    # For this example, assuming 'model.predict' handles it or you've pre-processed
    # input_df to match the format of xtrain.
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] # Probability for the positive class (>50K)
    except Exception as e:
        st.error(f"Error during prediction. Ensure input data is preprocessed correctly: {e}")
        st.stop()

    # Salary Result
    result_text = ">50K" if prediction == '>50K' else "≤50K" # Ensure this matches your model's output labels
    st.subheader(f"Predicted Income: **{result_text}**")

    # Personalized Advice
    if prediction == '>50K':
        st.success("🎉 You're on the higher earning side! Keep it up!")
    else:
        st.info("💡 Consider gaining experience, changing roles, or increasing working hours.")

    # Radar Chart for Profile
    st.markdown("#### Your Profile Radar")
    radar_fig = go.Figure()

    categories = ['Age', 'Hours/week', 'Education', 'Gain', 'Loss']
    # Normalize values for radar chart if they have vastly different scales
    # For simplicity, using raw values here, but consider min-max scaling for better visualization
    values = [age, hours, educational_num, capital_gain/1000 if capital_gain > 0 else 0, capital_loss/100 if capital_loss > 0 else 0]

    radar_fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Profile'
    ))

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=350, # Adjust height for better fit
        margin=dict(l=50, r=50, t=50, b=50) # Adjust margins
    )

    st.plotly_chart(radar_fig, use_container_width=True)

    # Gauge Chart (like a speedometer)
    st.markdown("#### Confidence Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]}, # Ensure range is 0 to 100 for percentage
            'bar': {'color': "green" if prediction == '>50K' else "red"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': { # Add a threshold line at 50%
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "Probability of >50K Salary"}
    ))
    gauge.update_layout(height=300, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(gauge, use_container_width=True)

# --- NEW UNIQUE SECTION: Model Performance Overview ---
st.markdown("---")
st.header("📊 Model Performance Overview")
st.write("This section provides a summary of the model's performance on the unseen test data.")

if 'xtest' in locals() and 'ytest' in locals(): # Check if test data was loaded successfully
    # Get predictions on the test set
    y_pred = model.predict(xtest)

    # 1. Overall Accuracy
    overall_accuracy = accuracy_score(ytest, y_pred)
    st.metric(label="Overall Model Accuracy", value=f"{overall_accuracy:.4f}", delta_color="off")
    st.write(f"The model correctly predicted income for approximately **{overall_accuracy*100:.2f}%** of the individuals in the test set.")

    # 2. Classification Report
    st.subheader("Detailed Classification Report")
    # Ensure target_names match your actual class labels (e.g., ['<=50K', '>50K'])
    class_report = classification_report(ytest, y_pred, target_names=['<=50K', '>50K'])
    st.text(class_report)
    st.write("""
    * **Precision:** Out of all predicted positives, how many were actually positive.
    * **Recall:** Out of all actual positives, how many were correctly predicted as positive.
    * **F1-Score:** The harmonic mean of precision and recall, a balanced metric.
    * **Support:** The number of occurrences of each class in `ytest`.
    """)

    # 3. Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(ytest, y_pred, labels=['<=50K', '>50K']) # Specify labels for correct order

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted <=50K', 'Predicted >50K'],
        y=['Actual <=50K', 'Actual >50K'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        hoverinfo='text'
    ))
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.write("""
    * **True Negatives (Top-Left):** Correctly predicted as ≤50K.
    * **False Positives (Top-Right):** Predicted as >50K, but actually ≤50K (Type I error).
    * **False Negatives (Bottom-Left):** Predicted as ≤50K, but actually >50K (Type II error).
    * **True Positives (Bottom-Right):** Correctly predicted as >50K.
    """)
else:
    st.warning("Model performance metrics cannot be displayed as test data was not loaded.")

# Footer
st.markdown("---")
st.caption("Made by Avanthika | Powered by Scikit-learn + Streamlit + Plotly")

