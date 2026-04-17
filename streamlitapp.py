import numpy as np
import pickle
import pandas as pd
import streamlit as st
import joblib

#loading model which was saved
# loaded_model = None

# try:
#     with open('trained_model.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# loading model
@st.cache_resource
def load_model():
    try:
        return joblib.load('trained_model.pkl')
    except Exception:
        return None

model = load_model()

def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except Exception:
        return None
    
scaler = load_scaler()


# making predictions
def loan_default_prediction(input_data):
    if model is None:
        return "Model not loaded"

    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[:]

    if pred == 0:
        return f"Likely to default (Risk: {prob[:, 0]})"
    else:
        return f"Less likely to default (Safe: {prob[:, 1]})"



def main():
    # setting the title
    st.title('Credit Risk Decision System App')
    
    # load my input data
    loan_purpose_house = st.text_input('Loan purpose if house. Yes: 1 No: 0', key='loan_purpose_house')
    loan_purpose_credit_card = st.text_input('Loan purpose if credit card. Yes: 1 No: 0', key='loan_purpose_credit_card')
    inquiries_last_12m = st.text_input('Inquiries made in the last 12 months', key='inquiries_last_12m')
    num_total_cc_accounts = st.text_input('Number of current accounts', key='num_total_cc_accounts')
    loan_amount = st.text_input('Loan amount', key='loan_amount')
    term = st.text_input('Loan term 36 or 60.', key='term')
    application_type_joint = st.text_input('Application Type: Joint. Yes: 1 No: 0', key='application_type_joint')

    #prediction variable
    prediction_string = ''

    #predicting loan default    
    if st.button('Loan Default Test Result'):
        inputs = [loan_purpose_house,loan_purpose_credit_card,inquiries_last_12m,num_total_cc_accounts,loan_amount, term, application_type_joint]

        if "" in inputs:
            st.error("Fill all fields")

        else:
            numeric_inputs = [float(i) for i in inputs]

            prediction_string = loan_default_prediction(numeric_inputs)
            st.success(prediction_string)



    # st.success(prediction_string)



if __name__ == '__main__':
    main()