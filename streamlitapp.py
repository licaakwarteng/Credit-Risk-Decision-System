import numpy as np
# from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import streamlit as st

#loading model which was saved
try:
    with open('trained_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    print("Error loading model:", e)

# making predictions
def loan_default_prediction(input_data):

    #load my input data
    input_data = pd.to_numeric(input_data)
    input_data = input_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data)
    print(prediction)
    if prediction[0] == 0:
        return 'Less likely to default'
    else:
        return 'Likely to default'



def main():
    # setting the title
    st.title('Credit Risk Decision System App')
    
    # load my input data

    income_per_credit_line = st.text_input('Income per credit line', key='income_per_credit_line')
    loan_purpose_house = st.text_input('Loan purpose if house. Yes: 1 No: 0', key='loan_purpose_house')
    loan_purpose_credit_card = st.text_input('Loan purpose if credit card. Yes: 1 No: 0', key='loan_purpose_credit_card')
    inquiries_last_12m = st.text_input('Inquiries made in the last 12 months', key='inquiries_last_12m')
    term = st.text_input('Loan term', key='term')
    num_total_cc_accounts = st.text_input('Number of current accounts', key='num_total_cc_accounts')
    months_since_last_credit_inquiry = st.text_input('Months since last credit inquiry', key='months_since_last_credit_inquiry')
    num_open_cc_accounts = st.text_input('Number of open accounts', key='num_open_cc_accounts')
    open_credit_lines = st.text_input('Number of open credit lines', key='open_credit_lines')
    num_satisfactory_accounts = st.text_input('Number of satisfactory accounts', key='num_satisfactory_accounts')
    loan_amount = st.text_input('Loan amount', key='loan_amount')
    application_type_joint = st.text_input('Application Type: Joint. Yes: 1 No: 0', key='application_type_joint')

    #prediction variable
    prediction_string = ''

    #predicting loan default    
    if st.button('Loan Default Test Result'):
        prediction_string = loan_default_prediction([income_per_credit_line,loan_purpose_house,loan_purpose_credit_card,inquiries_last_12m,term,num_total_cc_accounts,months_since_last_credit_inquiry,num_open_cc_accounts,open_credit_lines,num_satisfactory_accounts,loan_amount,application_type_joint])

    st.success(prediction_string)



if __name__ == '__main__':
    main()