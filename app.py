import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Churn Predictor App', layout='centered')

st.title("Customer Churn Predictor 🔮")
st.write("Enter the customer details below to predict the likelihood of them churning.")

# Load model and preprocessing
try:
    data = joblib.load('churn_model.pkl')
    model = data['model']
    scaler = data['scaler']
    encoders = data['encoders']
    categorical_cols = data['categorical_cols']
    features = data['features']
    
    with st.form("customer_form"):
        st.subheader("Customer Profile")
        input_data = {}
        
        cols = st.columns(3)
        for i, f in enumerate(features):
            col = cols[i % 3]
            with col:
                if f in categorical_cols:
                    options = list(encoders[f].classes_)
                    val = st.selectbox(f, options)
                    input_data[f] = val
                else:
                    # numeric
                    val = st.number_input(f, value=0.0)
                    input_data[f] = val
                
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            df_input = pd.DataFrame([input_data])
            # encode
            for c in categorical_cols:
                # Handle unseen labels by setting to the first or using a robust approach, we'll assume exact match
                df_input[c] = encoders[c].transform(df_input[c])
                
            # scale
            df_input_scaled = scaler.transform(df_input)
            
            # predict
            prediction = model.predict(df_input_scaled)[0]
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(df_input_scaled)[0][1]
                prob_text = f" (Confidence: {prob:.2%})"
            else:
                prob_text = ""
            
            if prediction == 1:
                st.error(f"⚠️ The customer is likely to **CHURN**!{prob_text}")
            else:
                st.success(f"✅ The customer is likely to **STAY**.{prob_text}")

except FileNotFoundError:
    st.error("Model file not found. Please train the model first by running `python train.py`")
