import streamlit as st
import joblib
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer

model = joblib.load(r"/contents/lr_model.pkl")

with open(r"/contents/label_encoder.json", "r") as f:
    label_encoder = json.load(f)

ordinal_encoder_number_employee = joblib.load(r"/contents/ordinal_encoder_no_employees.pkl")
ordinal_encoder_Age_cat = joblib.load(r"/contents/ordinal_encoder_Age_cat.pkl")

st.title("Mental Health Condition Predictor")


age = st.number_input('Age', min_value=18, max_value=100, value=25)
gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
country = st.selectbox('Country', options=[
    'United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France', 
    'Portugal', 'Netherlands', 'Switzerland', 'Poland', 'Australia', 
    'Germany', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica', 
    'Austria', 'Ireland', 'India', 'South Africa', 'Italy', 'Sweden', 
    'Colombia', 'Latvia', 'Romania', 'Belgium', 'New Zealand', 
    'Zimbabwe', 'Spain', 'Finland', 'Uruguay', 'Israel', 
    'Bosnia and Herzegovina', 'Hungary', 'Singapore', 'Japan', 
    'Nigeria', 'Croatia', 'Norway', 'Thailand', 'Denmark', 
    'Bahamas, The', 'Greece', 'Moldova', 'Georgia', 'China', 
    'Czech Republic', 'Philippines'
])
self_employed = st.selectbox('Self Employed', options=['Yes', 'No'])
family_history = st.selectbox('Family History', options=['No', 'Yes'])
work_interfere = st.selectbox('Work Interfere', options=['Often', 'Rarely', 'Never', 'Sometimes'])
no_employees = st.selectbox('Number of Employees', options=['1-5','6-25','26-100','100-500','500-1000','More than 1000'])
remote_work = st.selectbox('Remote Work', options=['No', 'Yes'])
tech_company = st.selectbox('Tech Company', options=['Yes', 'No'])
benefits = st.selectbox('Benefits', options=['Yes', "Don't know", 'No'])
care_options = st.selectbox('Care Options', options=['Not sure', 'No', 'Yes'])
wellness_program = st.selectbox('Wellness Program', options=['No', "Don't know", 'Yes'])
seek_help = st.selectbox('Seek Help', options=['Yes', "Don't know", 'No'])
anonymity = st.selectbox('Anonymity', options=['Yes', "Don't know", 'No'])
leave = st.selectbox('Leave', options=['Very easy', 'Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult'])
mental_health_consequence = st.selectbox('Mental Health Consequence', options=['No', 'Maybe', 'Yes'])
phys_health_consequence = st.selectbox('Physical Health Consequence', options=['No', 'Yes', 'Maybe'])
coworkers = st.selectbox('Coworkers', options=['Some of them', 'No', 'Yes'])
supervisor = st.selectbox('Supervisor', options=['Yes', 'No', 'Some of them'])
mental_health_interview = st.selectbox('Mental Health Interview', options=['No', 'Yes', 'Maybe'])
phys_health_interview = st.selectbox('Physical Health Interview', options=['Maybe', 'No', 'Yes'])
mental_vs_physical = st.selectbox('Mental vs Physical', options=['Yes', "Don't know", 'No'])
obs_consequence = st.selectbox('Observation Consequence', options=['No', 'Yes'])
comments = st.text_area('Comments')


if st.button("Predict Condition"):
    input_data = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence,
        "comments": comments
    }

    
    df = pd.DataFrame([input_data])
    def agecat(x):
        if x<=20:
            return 'teen'
        elif x>20 and x<=30:
            return 'adult'
        elif x>30 and x<=50:
            return 'middle age'
        else:
            return 'old'
    df['Age'] = df['Age'].apply(agecat)
    df["no_employees"] = ordinal_encoder_number_employee.transform(df[["no_employees"]])
    df["Age_cat"] = ordinal_encoder_Age_cat.transform(np.array(df[["Age"]])) 
    df.drop(columns=["Age"], inplace=True)  

    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    comment_embedding = embedding_model.encode([comments])


    comment_df = pd.DataFrame(comment_embedding)
    comment_df.columns = comment_df.columns.astype(str)
    df = df.drop(columns=["comments"]) 
    final_input = pd.concat([df, comment_df], axis=1)

    for feature in final_input.columns:
        if feature in label_encoder:
            le = label_encoder[feature]
            final_input[feature] = final_input[feature].map(le)
    prediction = model.predict(final_input)[0]
    condition = label_encoder.get(str(prediction), "Unknown Condition")


    if condition == "1":
        st.subheader("Predicted Condition: Treatment Required")
    else:
        st.subheader("Predicted Condition: No Treatment Required")
