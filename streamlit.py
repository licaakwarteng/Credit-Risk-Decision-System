import numpy as np
import pickle
import pandas as pd
import streamlit as st

#loading model which was saved
loaded_model = pickle.load(open('C:/Users/naaod/Downloads/loan_payback_analysis/trained_model.sav', 'rb'))

# making predictions
def loan_default_prediction(input_data):

    #load my input data
    input_data = input_data.values

    prediction = loaded_model.predict(input_data)
    if prediction[0] == 0:
        return 'Less likely to default'
    else:
        return 'Likely to default'



def main():
    # setting the title
    st.title('Credit Risk Decision System App')
    
    # load my input data

    income_per_credit_line = st.text_input('Income per credit line')
    loan_purpose_house = st.text_input('Loan purpose if house')
    loan_purpose_credit_card = st.text_input('Loan purpose if credit card')
    inquiries_last_12m = st.text_input('Inquiries made in the last 12 months')
    term = st.text_input('Loan term')
    num_total_cc_accounts = st.text_input('Number of current accounts')
    months_since_last_credit_inquiry = st.text_input('Months since last credit inquiry')
    num_open_cc_accounts = st.text_input('Number of current accounts')
    open_credit_lines = st.text_input('Number of open credit lines')
    num_satisfactory_accounts = st.text_input('Number of satisfactory accounts')
    loan_amount = st.text_input('Loan amount')
    application_type_joint = st.text_input('Application Type: Individual/ Joint')

    #prediction variable
    prediction = ''

    #predicting loan default    
    if st.button('Loan Default Test Result'):
        prediction = loan_default_prediction(income_per_credit_line,loan_purpose_house,loan_purpose_credit_card,inquiries_last_12m,term,num_total_cc_accounts,months_since_last_credit_inquiry,num_open_cc_accounts,open_credit_lines,num_satisfactory_accounts,loan_amount,application_type_joint)

    st.success(prediction)



    if __name__ == '__main__':
        main()