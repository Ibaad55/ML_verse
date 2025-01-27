#importing for the whole APP
import numpy as np
import streamlit as st
from pandas.io.common import file_path_to_url
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
import joblib



#importing for the STOCK PRICE
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates





#Analyzers Models
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model2 = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

#**EDA**
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


#analyzer models
import json
from Classifier import KNearestNeighbours
from bs4 import BeautifulSoup
import requests, io
import PIL.Image
from urllib.request import urlopen
with open('./Movie_data/movie_data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open('./Movie_data/movie_titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)
hdr = {'User-Agent': 'Mozilla/5.0'}


#Algorithm models
import seaborn as sns
import base64
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#Importing for MACHINE LEARNING
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from catboost import CatBoostClassifier
import io
from ml_utility import preprocess_data, evaluate_model



st.set_page_config(
    page_title="ML Verse",
    page_icon="👑",
    layout="wide"
)

disease_selection = None
Finance_model_selection = None

diabetes_clinics = [
    {
        "name": "Sadhana Proactive Clinic",
        "Doctor": "Dr. Kamini D. Lakhiani specializes in diabetes care and obesity Management",
        "address": "Plot No 451, Ground Floor, Dindoshilla Building, Opposite Khar Gymkhana Ground, Mumbai, MH 400052",
        "website": "https://maps.app.goo.gl/4soTbKu1D2Bwbq7y9",
        "contact": "+91 12345 67890"
    },
{
        "name": "Dr A Kapoor's Diabetes Control Clinic ",
        "address": "Parth Business Plaza, Zenith Multispeciality Hospital, Malad,Mumbai400064",
        "website": "https://maps.app.goo.gl/PotDC1UtUdr3DxQe9",
        "contact": "+91 98203 81015"
    },
{
        "name": "Dr A Kapoor's Diabetes Control Clinic ",
        "address": "Parth Business Plaza, Zenith Multispeciality Hospital, Malad,Mumbai: 400064",
        "website": "https://maps.app.goo.gl/9Kx2xV2mJWJWcca38",
        "contact": "+91 98201 17787"
    },
{
        "name": "Dr. Devyani Sankpal: Best Diabetologist in Mumbai ",
        "address": "SAGE Hospital Office no 2 & 3 Ground floor Aditya Heritage Building next to Rustom Ji Elenza, Malad West, Mumbai,400064",
        "website": "https://maps.app.goo.gl/hztxAEG4drUZkVWWA",
        "contact": "+91 98195 38665"
    },
]

heart_clinics = [
    {
        "name": "Asian Heart Institute",
        "address": "G / N, G Block BKC, Bandra Kurla Complex, Bandra East, Mumbai, Maharashtra 400051",
        "website": "https://maps.app.goo.gl/CuHmVfGMX2TM1uke7",
        "contact": "022 6698 6666"
    },
{
        "name": "Fortis Hospital Mulund",
        "address": "Mulund - Goregaon Link Rd, Nahur West,Industrial Area,Bhandup West, Mumbai,400078",
        "website": "https://maps.app.goo.gl/nttkZu9VNfN2uc7g8",
        "contact": "022 6884 6143"
    },
{
        "name": "Kokilaben Dhirubhai Ambani Hospital and Medical Research Institute",
        "address": "Rao Saheb, Rao Saheb Achutrao Patwardhan Marg, Four Bungalows, Andheri[W},Mumbai:400053",
        "website": "http://www.heartcareclinic.com",
        "contact": " 022 4269 6969"
    },
{
        "name": "Lilavati Hospital and Research Centre",
        "address": "A-791, A-791, Bandra Reclamation Rd, General Arunkumar Vaidya Nagar, Bandra[W],Mumbai,400050",
        "website": "https://maps.app.goo.gl/LQiPvm18CvUWUMHm6",
        "contact": "022 6930 1000"
    },
]

parkinsons_clinics = [
    {
        "name": "EN1 Neuro Services Private Limited",
        "address": "D-510 Kanakia Zillion, 5th floor D-Wing Lal Bahadur Shastri Marg Kurla[W], Maharashtra 400070",
        "website": "https://maps.app.goo.gl/Q6QXuyS7t2EfsYgq8",
        "contact": "+91 9967272308"
    },
{
        "name": "P.D. Hinduja National Hospital & Medical Research Centre",
        "address": "8-12, Swatantryaveer Savarkar Rd, Mahim[W],Mahim, Mumbai,400016",
        "website": "https://maps.app.goo.gl/QZH2o9XX5NzDJCXJ6",
        "contact": "+91 2269248000"
    },
{
        "name": "Fortis Hospital Mulund",
        "address": "Mulund - Goregaon Link Rd, Nahur West,Industrial Area,Bhandup West, Mumbai,400078",
        "website": "https://maps.app.goo.gl/nttkZu9VNfN2uc7g8",
        "contact": "022 6884 6143"
    },

]


diabetes_model = pickle.load(open('I:/MultipleDiseasePrediction/mdpss/saved_models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('I:/MultipleDiseasePrediction/mdpss/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('I:/MultipleDiseasePrediction/mdpss/saved_models/parkinsons_model.sav', 'rb'))





#price predictions
house_prediction_model = joblib.load('saved_models/Heart_pred.sav')
Bank_Loan_model = pickle.load(open('I:\TEST_GODFATHER_PREDICTEOR\Papa_prediction\mdpss\LOAN_PREDICT_Model.pkl', 'rb'))

#fake new
vectorizer = joblib.load(r"I:\TEST_GODFATHER_PREDICTEOR\Papa_prediction\mdpss\saved_models\vectorizer.jb")
model = joblib.load(r"I:\TEST_GODFATHER_PREDICTEOR\Papa_prediction\mdpss\saved_models\lr_model.jb")

credit_fraud_model = pickle.load(open('I:\TEST_GODFATHER_PREDICTEOR\Papa_prediction\mdpss\saved_models\Credit_card_model.pkl', 'rb'))









# Sidebar menu with nested sections
with (st.sidebar):
    selected_section = option_menu(
        menu_title="𝕋𝕙𝕖 𝕄𝕃 𝕍𝕖𝕣𝕤𝕖",
        options=["About Project", "Diseases Prediction", "Finance Models","Fake Predictions","Analyzer Models","Algorithm Master"],
        icons=["book", "book", "book","book","book","book"],
        menu_icon="list-task",
        default_index=0
    )

    # Submenu for "Diseases Prediction"
    if selected_section == "Diseases Prediction":
        disease_selection = option_menu(
            menu_title="Choose Disease",
            options=["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
            icons=["clipboard-pulse", "chat-square-heart-fill", "file-earmark-ppt-fill"],
            menu_icon="activity",
            default_index=0
        )

    # Submenu for "Other Models"
    elif selected_section == "Finance Models":
        Finance_model_selection = option_menu(
            menu_title="Choose Prediction System",
            options=["House Price Prediction", "Stocks Visual Analyzer", "Stock Price Prediction","Bank Loan Prediction"],
            icons=["bar-chart", "bar-chart-line", "bar-chart-fill"],
            menu_icon="bar-chart-line",
            default_index=0
        )

    elif selected_section == "Fake Predictions":
        Fake_Models = option_menu(
            menu_title="Choose Model",
            options=["Fake News Prediction", "Credit Card Fraud Detection"],
            icons=["bar-chart", "bar-chart-line"],
            menu_icon="bar-chart-line",
            default_index=0
        )
    elif selected_section == "Analyzer Models":
        Analyzer_selection = option_menu(
            menu_title="Choose Prediction System",
            options=["Simple Sentiment Analysis", "Exploartory Data Analysis"],
            icons=["bar-chart", "bar-chart-line"],
            menu_icon="bar-chart-line",
            default_index=0
        )

    elif selected_section == "Algorithm Master":
        Algorithm_selection = option_menu(
            menu_title="Choose Prediction System",
            options=["AutoML", "No Code Machine Learning Trainer"],
            icons=["bar-chart", "bar-chart-line"],
            menu_icon="bar-chart-line",
            default_index=0
        )

# Display content based on sidebar selection
if selected_section == "About Project":
    st.markdown("<h1 style='text-align: center;font_size : 22px ;text-decoration: underline ; '>Ｔｈｅ ＭＬ Ｖｅｒｓｅ</h1>",
                unsafe_allow_html=True)

    image = Image.open('ai_IMG.jpg')
    st.image(image, caption='AI GENERATED', width=1200)
    st.markdown("<h1 style='text-align: center;font_size : 22px ;text-decoration: underline ; '>𝐀𝐁𝐎𝐔𝐓 𝐌𝐋 𝐕𝐄𝐑𝐒𝐄 :</h1>",
                unsafe_allow_html=True)

    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            line-height: 1.6;
        }
        </style>
        """, unsafe_allow_html=True)

    with open('about.txt', 'r') as file:
        info = file.read()
        st.markdown(f'<div class="big-font">{info}</div>', unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center;font_size : 22px ;text-decoration: underline ; '>Take a leap into the future of AI-powered insights. Unleash the potential of your data with The ML Verse!</h1>",
            unsafe_allow_html=True)






elif selected_section == "Diseases Prediction":
    if disease_selection == "Diabetes Prediction":
        st.markdown("<h1 style='text-align: center;text-decoration: underline ;color : #FF7074; '>DIABETES PREDICTION SYSTEM</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; '>This section allows you to predict the likelihood of diabetes</h3>",
                    unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; text-decoration: underline;'>𝗜𝗺𝗽𝗼𝗿𝘁𝗮𝗻𝘁 𝗡𝗼𝘁𝗶𝗰𝗲</h2>",
                    unsafe_allow_html=True)
        with open('diabetes.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)

        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')

        with col2:
            Glucose = st.text_input('Glucose Level')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value')

        with col2:
            Insulin = st.text_input('Insulin Level')

        with col3:
            BMI = st.text_input('BMI value')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

        with col2:
            Age = st.text_input('Age of the Person')

        diab_diagnosis = ''  # Initialize the result variable

        # Add the missing code here to show the prediction result
        if st.button('Diabetes Test Result'):
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                          Age]

            if any(x == '' for x in user_input):
                st.error('Please enter all the values.')
            else:
                try:
                    user_input = [float(x) for x in user_input]
                    diab_prediction = diabetes_model.predict([user_input])

                    if diab_prediction[0] == 1:
                        diab_diagnosis = 'THE PERSON IS DIABETIC'
                        st.markdown(
                            f"""
                            <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #f0fff0;'>
                                <h2 style='color: green; margin: 0;'>{diab_diagnosis}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.write("BEST Diabetic Clinics in Mumbai:")
                        for clinic in diabetes_clinics:
                            st.write(f"**{clinic['name']}** - {clinic['address']}  \n"
                                     f"Website: [Link]({clinic['website']})  \n"
                                     f"Contact: {clinic['contact']}")
                    else:
                        diab_diagnosis = 'THE PERSON IS NOT DIABETIC'
                        st.markdown(
                            f"""
                            <div style='text-align: center; border: 2px solid red; border-radius: 10px; padding: 10px; background-color: #fff0f0;'>
                                <h2 style='color: red; margin: 0;'>{diab_diagnosis}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        with open('precautions_for_non_diabetic_individuals.txt', 'r') as file:
                            info = file.read()
                            st.write("")
                            st.write(info)
                except ValueError:
                    st.error('Please enter valid numeric values.')






    elif disease_selection == "Heart Disease Prediction":
        st.markdown("<h1 style='text-align: center;text-decoration: underline ;color : #FF7074; '>HEART DISEASE PREDICTION SYSTEM</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; text-decoration: underline;'>𝗜𝗺𝗽𝗼𝗿𝘁𝗮𝗻𝘁 𝗡𝗼𝘁𝗶𝗰𝗲</h2>",
                    unsafe_allow_html=True)
        with open('heart_se.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)
        st.title('Heart Disease Prediction using ML')

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age of the Person')

        with col2:
            sex = st.text_input('Sex (0 = Female, 1 = Male)')

        with col3:
            cp = st.text_input('Chest Pain Type (0-3)')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure')

        with col2:
            chol = st.text_input('Cholesterol Level')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar (1 = True, 0 = False)')

        with col1:
            restecg = st.text_input('Resting Electrocardiographic Results (0-2)')

        with col2:
            thalach = st.text_input('Maximum Heart Rate Achieved')

        with col3:
            exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')

        with col1:
            oldpeak = st.text_input('Depression Induced by Exercise Relative to Rest')

        with col2:
            slope = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')

        with col3:
            ca = st.text_input('Number of Major Vessels (0-3)')

        with col1:
            thal = st.text_input('Thalassemia (0-3)')

        heart_diagnosis = ''

        if st.button('Heart Disease Test Result'):
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            if any(x == '' for x in user_input):
                st.error('Please enter all the values.')
            else:
                try:
                    user_input = [float(x) for x in user_input]
                    heart_prediction = heart_model.predict([user_input])

                    if heart_prediction[0] == 1:
                        heart_diagnosis = 'THE PERSON HAS HEART DISEASE'
                        st.markdown(
                            f"""
                                <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #f0fff0;'>
                                    <h2 style='color: green; margin: 0;'>{heart_diagnosis}</h2>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                        st.write("Nearby Heart Clinics in Mumbai:")
                        for clinic in heart_clinics:
                            st.write(f"**{clinic['name']}** - {clinic['address']}  \n"
                                     f"Website: [Link]({clinic['website']})  \n"
                                     f"Contact: {clinic['contact']}")
                    else:
                        heart_diagnosis = 'THE PERSON DOES NOT HAVE HEART DISEASE'
                        st.markdown(
                            f"""
                                <div style='text-align: center; border: 2px solid red; border-radius: 10px; padding: 10px; background-color: #fff0f0;'>
                                    <h2 style='color: red; margin: 0;'>{heart_diagnosis}</h2>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                        with open('precautions_for_heart_disease.txt', 'r') as file:
                            info = file.read()
                            st.write("")
                            st.write(info)
                except ValueError:
                    st.error('Please enter valid numeric values.')


    # Add your heart disease prediction model code here

    elif disease_selection == "Parkinson's Prediction":
        st.markdown("<h1 style='text-align: center;text-decoration: underline ;color : #FF7074; '>PARKINSON'S PREDICTION SYSTEM</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; text-decoration: underline;'>𝗜𝗺𝗽𝗼𝗿𝘁𝗮𝗻𝘁 𝗡𝗼𝘁𝗶𝗰𝗲</h2>",
                    unsafe_allow_html=True)
        with open('parkkin_se.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')

        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')

        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')

        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')

        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

        with col1:
            RAP = st.text_input('MDVP:RAP')

        with col2:
            PPQ = st.text_input('MDVP:PPQ')

        with col3:
            DDP = st.text_input('Jitter:DDP')

        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')

        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')

        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')

        with col3:
            APQ = st.text_input('MDVP:APQ')

        with col4:
            DDA = st.text_input('Shimmer:DDA')

        with col5:
            NHR = st.text_input('NHR')

        with col1:
            HNR = st.text_input('HNR')

        with col2:
            RPDE = st.text_input('RPDE')

        with col3:
            DFA = st.text_input('DFA')

        with col4:
            spread1 = st.text_input('spread1')

        with col5:
            spread2 = st.text_input('spread2')

        with col1:
            D2 = st.text_input('D2')

        with col2:
            PPE = st.text_input('PPE')

        parkinsons_diagnosis = ''

        if st.button("Parkinson's Test Result"):
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            if any(x == '' for x in user_input):
                st.error('Please enter all the values.')
            else:
                try:
                    user_input = [float(x) for x in user_input]
                    parkinsons_prediction = parkinsons_model.predict([user_input])

                    if parkinsons_prediction[0] == 1:
                        parkinsons_diagnosis = "THE PERSON HAS PARKINSON'S DISEASE"
                        st.markdown(
                            f"""
                                <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #f0fff0;'>
                                    <h2 style='color: green; margin: 0;'>{parkinsons_diagnosis}</h2>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                        st.write("Nearby Parkinson's Clinics in Mumbai:")
                        for clinic in parkinsons_clinics:
                            st.write(f"**{clinic['name']}** - {clinic['address']}  \n"
                                     f"Website: [Link]({clinic['website']})  \n"
                                     f"Contact: {clinic['contact']}")
                    else:
                        parkinsons_diagnosis = "THE PERSON DOES NOT HAVE PARKINSON'S DISEASE"
                        st.markdown(
                            f"""
                                <div style='text-align: center; border: 2px solid red; border-radius: 10px; padding: 10px; background-color: #fff0f0;'>
                                    <h2 style='color: red; margin: 0;'>{parkinsons_diagnosis}</h2>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )
                        with open('precautions_for_parkinsons.txt.txt', 'r') as file:
                            info = file.read()
                            st.write("")
                            st.write(info)
                except ValueError:
                    st.error('Please enter valid numeric values.')

elif selected_section == "Finance Models":
    if Finance_model_selection== "House Price Prediction":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>House Price Prediction</h1>",
                    unsafe_allow_html=True)
        with open('caution.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)
            st.markdown("<h4 style='text-align: left; text-decoration: underline;'>User Inputs for House Price Prediction : </h4>",
                        unsafe_allow_html=True)
            st.markdown(
                "<h4 style='text-align: left;'>To make predictions regarding House Price, we require the following inputs from the user:</h5>",
                unsafe_allow_html=True)
            with open('house_UI.txt', 'r') as file:
                info = file.read()
                st.write("")
                st.write(info)


        col1, col2, col3 = st.columns(3)
        with col1:
            longitute = st.number_input('Enter Longitute of house',min_value=0)

        with col2:
            latitude = st.number_input('Enter Latitude of house',min_value=0)

        with col3:
            age = st.number_input('How old is the house (in years)?', min_value=0, step=1)

        with col1:
            rooms = st.number_input('Total number of rooms',min_value=0)

        with col2:
            median_income = st.number_input('Median income of house',min_value=0)

        with col3:
            ocean_proximity = st.selectbox(
        'Specify Ocean proximity ?',
        ('Choose your Option', '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))
        if ocean_proximity == '<1H OCEAN':
            OCEAN = 1.0
            INLAND = 0.0
            ISLAND = 0.0
            NEAR_BAY = 0.0
            NEAR_OCEAN = 0.0
        elif ocean_proximity == 'INLAND':
            OCEAN = 0.0
            INLAND = 1.0
            ISLAND = 0.0
            NEAR_BAY = 0.0
            NEAR_OCEAN = 0.0
        elif ocean_proximity == 'ISLAND':
            OCEAN = 0.0
            INLAND = 0.0
            ISLAND = 1.0
            NEAR_BAY = 0.0
            NEAR_OCEAN = 0.0
        elif ocean_proximity == 'NEAR BAY':
            OCEAN = 0.0
            INLAND = 0.0
            ISLAND = 0.0
            NEAR_BAY = 1.0
            NEAR_OCEAN = 0.0
        elif ocean_proximity == 'NEAR OCEAN':
            OCEAN = 0.0
            INLAND = 0.0
            ISLAND = 0.0
            NEAR_BAY = 0.0
            NEAR_OCEAN = 1.0

        with col1:
            bedroom_ratio = st.text_input('Total number of bedrooms ')

        with col2:
            household_rooms = st.number_input('Total number of rooms per household')
        if st.button('Predict House Price'):
            cost = house_prediction_model.predict(
                np.array([[longitute, latitude, age, rooms, median_income, OCEAN, INLAND, ISLAND, NEAR_BAY,
                           NEAR_OCEAN, bedroom_ratio, household_rooms]]))
            st.markdown("<h1 style='text-align: center; text-decoration: underline;'>Predicted House Price is in $</h1>",
                        unsafe_allow_html=True)
            st.markdown(
                f"""
                                    <div style='text-align: center; border: 2px solid green; border-radius: 10px; padding: 10px; background-color: #f0fff0;'>
                                        <h2 style='color: green; margin: 0;'>{cost}</h2>
                                    </div>
                                    """,
                unsafe_allow_html=True
            )

    elif Finance_model_selection == "Stocks Visual Analyzer":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'> STOCKS ANALYZER</h1>",unsafe_allow_html=True)
        st.title("AI-Powered Technical Stock Analysis Dashboard")
        with open('caution.txt','r') as file:
            info = file.read()
            st.write("")
            st.write(info)
            st.markdown(
                "<h4 style='text-align: left; text-decoration: underline;'>User Inputs for Stocks Visual Analyzer : </h4>",
                unsafe_allow_html=True)
            st.markdown(
                "<h4 style='text-align: left;'>To Visualize the Stock Prices, we require the following inputs from the user:</h5>",
                unsafe_allow_html=True)
        file_path_sta ='I:\TEST_GODFATHER_PREDICTEOR\Papa_prediction\mdpss\saved_models\lol.txt'
        with open(file_path_sta) as file:
            info = file.read()
            st.write("")
            st.write(info)


        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-14"))

        # Fetch stock data
        if st.button("Fetch Data"):
            st.session_state["stock_data"] = yf.download(ticker, start=start_date, end=end_date)
            st.success("Stock data loaded successfully!")

        # Check if data is available
        if "stock_data" in st.session_state:
            data = st.session_state["stock_data"]

            # Plot candlestick chart
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="Candlestick"  # Replace "trace 0" with "Candlestick"
                )
            ])

            # Sidebar: Select technical indicators
            st.sidebar.subheader("Technical Indicators")
            indicators = st.sidebar.multiselect(
                "Select Indicators:",
                ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
                default=["20-Day SMA"]
            )


            # Helper function to add indicators to the chart
            def add_indicator(indicator):
                if indicator == "20-Day SMA":
                    sma = data['Close'].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
                elif indicator == "20-Day EMA":
                    ema = data['Close'].ewm(span=20).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
                elif indicator == "20-Day Bollinger Bands":
                    sma = data['Close'].rolling(window=20).mean()
                    std = data['Close'].rolling(window=20).std()
                    bb_upper = sma + 2 * std
                    bb_lower = sma - 2 * std
                    fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                    fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
                elif indicator == "VWAP":
                    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))


            # Add selected indicators to the chart
            for indicator in indicators:
                add_indicator(indicator)

            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)




    elif Finance_model_selection== "Stock Price Prediction":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>Stock Price Prediction</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Enhanced Stock Price Predictor Prediction</h1>",
                    unsafe_allow_html=True)
        with open('caution.txt','r') as file:
            info = file.read()
            st.write("")
            st.write(info)
        st.markdown(
            "<h4 style='text-align: left; text-decoration: underline;'>User Inputs for Stock Price Prediction : </h4>",
            unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: left;'>To make predictions regarding Stock Price, we require the following inputs from the user:</h5>",
            unsafe_allow_html=True)
        with open('stock_price_ui.txt','r') as file:
            info = file.read()
            st.write("")
            st.write(info)


        stock = st.text_input('Enter Stock Symbol', 'GOOG')
        start_date = st.date_input('Start Date', value=pd.to_datetime('2012-01-01'))
        end_date = st.date_input('End Date', value=pd.to_datetime('2022-12-31'))

        # Button to fetch data
        if st.button('Fetch Data'):
            # Download stock data
            data = yf.download(stock, start=start_date, end=end_date)

            if not data.empty:
                st.subheader('Stock Data')
                st.write(data)

                # Prepare the data for linear regression
                data['Date'] = data.index
                data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.timestamp)  # Convert to timestamp
                X = data[['Date']]  # Features
                y = data['Close']  # Target variable

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create and train the model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_test)

                # Plot the results
                st.subheader('Predictions vs Actual Prices')
                fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
                ax.plot(y_test.index, y_test, label='Actual Prices', color='blue', linewidth=2)
                ax.plot(y_test.index, predictions, label='Predicted Prices', color='red', linestyle='--', linewidth=2)

                # Formatting the plot
                ax.set_xlabel('Date', fontsize=14)
                ax.set_ylabel('Price', fontsize=14)
                ax.set_title(f'Stock Price Prediction for {stock}', fontsize=16)
                ax.legend()
                ax.grid(True)  # Add gridlines

                # Format the x-axis to show dates nicely
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  # Rotate date labels

                st.pyplot(fig)

                # Moving Averages
                st.subheader('Moving Averages')
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()

                fig_ma, ax_ma = plt.subplots(figsize=(12, 6))
                ax_ma.plot(data['Close'], label='Close Price', color='blue')
                ax_ma.plot(data['MA50'], label='50-Day MA', color='orange')
                ax_ma.plot(data['MA200'], label='200-Day MA', color='green')
                ax_ma.set_title(f'Moving Averages for {stock}', fontsize=16)
                ax_ma.set_xlabel('Date', fontsize=14)
                ax_ma.set_ylabel('Price', fontsize=14)
                ax_ma.legend()
                ax_ma.grid(True)
                st.pyplot(fig_ma)

                # Volume Bar Chart
                st.subheader('Trading Volume')
                fig_vol, ax_vol = plt.subplots(figsize=(12, 6))
                ax_vol.bar(data.index, data['Volume'], color='purple', alpha=0.6)
                ax_vol.set_title(f'Trading Volume for {stock}', fontsize=16)
                ax_vol.set_xlabel('Date', fontsize=14)
                ax_vol.set_ylabel('Volume', fontsize=14)
                st.pyplot(fig_vol)

                # Show model performance
                st.subheader('Model Performance')
                st.write(f'Mean Absolute Error: {np.mean(np.abs(predictions - y_test)):.2f}')
            else:
                st.error("No data found for the given stock symbol and date range.")


    elif Finance_model_selection == "Bank Loan Prediction":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>Bank Loan Prediction</h1>",unsafe_allow_html=True)


        def run():
            img1 = Image.open('bank.jpg')
            img1 = img1.resize((1100, 600))
            st.image(img1, use_container_width=False)
            st.title("Bank Loan Prediction using Machine Learning")
            with open('caution.txt','r') as file:
                info = file.read()
                st.write(info)
            st.markdown(
                "<h4 style='text-align: left; text-decoration: underline;'>User Inputs for Bank Loan Prediction : </h4>",
                unsafe_allow_html=True)
            st.markdown(
                "<h4 style='text-align: left;'>To make predictions regarding Loan Approval, we require the following inputs from the user:</h5>",
                unsafe_allow_html=True)
            with open('bank_loan_pred.txt', 'r') as file:
                info = file.read()
                st.write(info)

            account_no = st.text_input('Account number')

            fn = st.text_input('Full Name')

            gen_display = ('Female', 'Male')
            gen_options = list(range(len(gen_display)))
            gen = st.selectbox("Gender", gen_options, format_func=lambda x: gen_display[x])

            mar_display = ('No', 'Yes')
            mar_options = list(range(len(mar_display)))
            mar = st.selectbox("Marital Status", mar_options, format_func=lambda x: mar_display[x])

            dep_display = ('No', 'One', 'Two', 'More than Two')
            dep_options = list(range(len(dep_display)))
            dep = st.selectbox("Dependents", dep_options, format_func=lambda x: dep_display[x])

            edu_display = ('Not Graduate', 'Graduate')
            edu_options = list(range(len(edu_display)))
            edu = st.selectbox("Education", edu_options, format_func=lambda x: edu_display[x])

            emp_display = ('Job', 'Business')
            emp_options = list(range(len(emp_display)))
            emp = st.selectbox("Employment Status", emp_options, format_func=lambda x: emp_display[x])

            prop_display = ('Rural', 'Semi-Urban', 'Urban')
            prop_options = list(range(len(prop_display)))
            prop = st.selectbox("Property Area", prop_options, format_func=lambda x: prop_display[x])

            cred_display = ('Between 300 to 500', 'Above 500')
            cred_options = list(range(len(cred_display)))
            cred = st.selectbox("Credit Score", cred_options, format_func=lambda x: cred_display[x])

            mon_income = st.number_input("Applicant's Monthly Income($)", value=0)

            co_mon_income = st.number_input("Co-Applicant's Monthly Income($)", value=0)

            loan_amt = st.number_input("Loan Amount", value=0)

            dur_display = ['2 Month', '6 Month', '8 Month', '1 Year', '16 Month']
            dur_options = range(len(dur_display))
            dur = st.selectbox("Loan Duration", dur_options, format_func=lambda x: dur_display[x])

            if st.button("Submit"):
                duration = 0
                if dur == 0:
                    duration = 60
                if dur == 1:
                    duration = 180
                if dur == 2:
                    duration = 240
                if dur == 3:
                    duration = 360
                if dur == 4:
                    duration = 480
                features = [[gen, mar, dep, edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
                print(features)
                prediction = Bank_Loan_model.predict(features)
                lc = [str(i) for i in prediction]
                ans = int("".join(lc))
                if ans == 0:
                    st.error(
                            "Hello: " + fn + " || "
                            "Account number: " + account_no + ' || '
                            'According to our Calculations, you will not get the loan from Bank'
                            )
                else:
                    st.success(
                            "Hello: " + fn + " || "
                            "Account number: " + account_no + ' || '
                            'Congratulations!! you will get the loan from Bank'
                            )


        run()






elif selected_section == "Fake Predictions":
    if Fake_Models == "Fake News Prediction":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>Fake News Prediction</h1>",
                    unsafe_allow_html=True)
        with open('caution.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)
        st.markdown("<h3 style='text-align: center; '> ↓ Enter a News Article below to check whether it is Fake or Real. ↓ </h3>",
                    unsafe_allow_html=True)
        inputn = st.text_area("News Article:", "")

        if st.button("Check News"):
            if inputn.strip():
                transform_input = vectorizer.transform([inputn])
                prediction = model.predict(transform_input)

                if prediction[0] == 1:
                    st.markdown(
                        f"""
                        <div style='
                            text-align: center; 
                            border: 3px solid #6D28D9; 
                            border-radius: 20px; 
                            padding: 25px; 
                            background: linear-gradient(145deg, #4C1D95, #5B21B6); 
                            box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                            font-family: "Trebuchet MS", sans-serif;
                            position: relative;
                            overflow: hidden;
                            transform: perspective(1000px);'>
                            <div style='
                                position: absolute;
                                top: -50%;
                                left: -50%;
                                width: 200%;
                                height: 200%;
                                background: 
                                    radial-gradient(circle at center, 
                                        rgba(109, 40, 217, 0.1) 0%, 
                                        transparent 70%);
                                opacity: 0.5;
                                z-index: 1;
                            '></div>
                            <div style='
                                position: relative; 
                                z-index: 2;
                                transform: rotateX(5deg);'>
                                <h2 style='
                                    color: #E9D5FF; 
                                    margin: 0; 
                                    font-size: 34px; 
                                    font-weight: 800;
                                    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
                                    letter-spacing: 3px;'>
                                    {"THE NEWS IS REAL"}
                                </h2>
                                <p style='
                                    color: #D8B4FE; 
                                    font-size: 19px; 
                                    margin-top: 15px;
                                    font-weight: 700;
                                    text-transform: uppercase;
                                    letter-spacing: 2px;'>
                                    Misinformation Detected
                                </p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.success("The News is Real! ")
                else:
                    st.markdown(
                        f"""
                        <div style='
                            text-align: center; 
                            border: 3px solid #6D28D9; 
                            border-radius: 20px; 
                            padding: 25px; 
                            background: linear-gradient(145deg, #4C1D95, #5B21B6); 
                            box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                            font-family: "Trebuchet MS", sans-serif;
                            position: relative;
                            overflow: hidden;
                            transform: perspective(1000px);'>
                            <div style='
                                position: absolute;
                                top: -50%;
                                left: -50%;
                                width: 200%;
                                height: 200%;
                                background: 
                                    radial-gradient(circle at center, 
                                        rgba(109, 40, 217, 0.1) 0%, 
                                        transparent 70%);
                                opacity: 0.5;
                                z-index: 1;
                            '></div>
                            <div style='
                                position: relative; 
                                z-index: 2;
                                transform: rotateX(5deg);'>
                                <h2 style='
                                    color: #E9D5FF; 
                                    margin: 0; 
                                    font-size: 34px; 
                                    font-weight: 800;
                                    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
                                    letter-spacing: 3px;'>
                                    {"THE NEWS IS FAKE"}
                                </h2>
                                <p style='
                                    color: #D8B4FE; 
                                    font-size: 19px; 
                                    margin-top: 15px;
                                    font-weight: 700;
                                    text-transform: uppercase;
                                    letter-spacing: 2px;'>
                                    Misinformation Detected
                                </p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.warning("Please enter some text to Analyze. ")



    elif Fake_Models == "Credit Card Fraud Detection":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;'>Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
        with open('caution.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)
        st.markdown(
            "<h4 style='text-align: left; text-decoration: underline;'>User Inputs for Credit Card Fraud Detection : </h4>",
            unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: left;'>To make predictions regarding Credit card, we require the following inputs from the user:</h5>",
            unsafe_allow_html=True)
        with open('credit_card_fraud_detection.txt', 'r') as file:
            info = file.read()
            st.write("")
            st.write(info)

        col1, col2, col3 = st.columns(3)

        with col1:
            time = st.number_input("Time (in seconds)", min_value=0)
            V1 = st.number_input("V1", min_value=0.0, max_value=100.0)
            V2 = st.number_input("V2", min_value=0.0, max_value=100.0)
            V3 = st.number_input("V3", min_value=0.0, max_value=100.0)
            V4 = st.number_input("V4", min_value=0.0, max_value=100.0)
            V5 = st.number_input("V5", min_value=0.0, max_value=100.0)
            V6 = st.number_input("V6", min_value=0.0, max_value=100.0)
            V7 = st.number_input("V7", min_value=0.0, max_value=100.0)
            V8 = st.number_input("V8", min_value=0.0, max_value=100.0)
            V9 = st.number_input("V9", min_value=0.0, max_value=100.0)

        with col2:
            V10 = st.number_input("V10", min_value=0.0, max_value=100.0)
            V11 = st.number_input("V11", min_value=0.0, max_value=100.0)
            V12 = st.number_input("V12", min_value=0.0, max_value=100.0)
            V13 = st.number_input("V13", min_value=0.0, max_value=100.0)
            V14 = st.number_input("V14", min_value=0.0, max_value=100.0)
            V15 = st.number_input("V15", min_value=0.0, max_value=100.0)
            V16 = st.number_input("V16", min_value=0.0, max_value=100.0)
            V17 = st.number_input("V17", min_value=0.0, max_value=100.0)
            V18 = st.number_input("V18", min_value=0.0, max_value=100.0)

        with col3:
            V19 = st.number_input("V19", min_value=0.0, max_value=100.0)
            V20 = st.number_input("V20", min_value=0.0, max_value=100.0)
            V21 = st.number_input("V21", min_value=0.0, max_value=100.0)
            V22 = st.number_input("V22", min_value=0.0, max_value=100.0)
            V23 = st.number_input("V23", min_value=0.0, max_value=100.0)
            V24 = st.number_input("V24", min_value=0.0, max_value=100.0)
            V25 = st.number_input("V25", min_value=0.0, max_value=100.0)
            V26 = st.number_input("V26", min_value=0.0, max_value=100.0)
            V27 = st.number_input("V27", min_value=0.0, max_value=100.0)
            V28 = st.number_input("V28", min_value=0.0, max_value=100.0)
            amount = st.number_input("Amount", min_value=0.0, max_value=1000000.0)

        input_data = {
            'Time': time,
            'V1': V1,
            'V2': V2,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'V6': V6,
            'V7': V7,
            'V8': V8,
            'V9': V9,
            'V10': V10,
            'V11': V11,
            'V12': V12,
            'V13': V13,
            'V14': V14,
            'V15': V15,
            'V16': V16,
            'V17': V17,
            'V18': V18,
            'V19': V19,
            'V20': V20,
            'V21': V21,
            'V22': V22,
            'V23': V23,
            'V24': V24,
            'V25': V25,
            'V26': V26,
            'V27': V27,
            'V28': V28,
            'Amount': amount
        }

        if st.button("Predict Fraud"):
            input_df = pd.DataFrame([input_data])
            prediction = credit_fraud_model.predict(input_df)

            if any(x == '' for x in input_data):
                st.error('Please enter all the values.')
            else:
                try:
                    if prediction[0] == 1:
                        st.markdown(
                                    f"""
                                <div style='
                                text-align: center; border: 3px solid #6D28D9; border-radius: 15px; padding: 25px; background: linear-gradient(145deg, #4C1D95, #5B21B6); box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                                font-family: "Bold Addict", sans-serif; position: relative; overflow: hidden; transform: perspective(1000px);'><div style='
                                position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                                background: radial-gradient(circle at center, rgba(109, 40, 217, 0.1) 0%, transparent 70%);
                                opacity: 0.5; z-index: 1;'></div><div style='position: relative; z-index: 2; transform: rotateX(5deg);'>
                                <h2 style='color: #E9D5FF; margin: 0; font-size: 34px; font-weight: 800;text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 3px;'>
                                {"Fraudulent Transaction Detected!"}
                                </h2></div></div>
                                """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.markdown(
                                f"""
                                    <div style='
                                    text-align: center; border: 3px solid #6D28D9; border-radius: 15px; padding: 25px; background: linear-gradient(145deg, #4C1D95, #5B21B6); box-shadow: 0 12px 40px rgba(109, 40, 217, 0.3);
                                    font-family: "Bold Addict", sans-serif; position: relative; overflow: hidden; transform: perspective(1000px);'><div style='
                                    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
                                    background: radial-gradient(circle at center, rgba(109, 40, 217, 0.1) 0%, transparent 70%);
                                    opacity: 0.5; z-index: 1;'></div><div style='position: relative; z-index: 2; transform: rotateX(5deg);'>
                                    <h2 style='color: #E9D5FF; margin: 0; font-size: 34px; font-weight: 800;text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 3px;'>
                                    {"Transaction is Legitimate"}
                                    </h2></div></div>
                                    """,
                                unsafe_allow_html=True
                            )
                except ValueError:
                    st.error('Please enter valid numeric values.')


elif selected_section == "Analyzer Models":
    if Analyzer_selection == "Simple Sentiment Analysis":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>🕊️ Simple Tweets Sentiment Analysis 🕊️</h1>",unsafe_allow_html=True)
        labels = ['Negative 😞', 'Neutral 😐', 'Positive 😀']

        st.markdown("<h4 style='text-align: center;text-decoration: underline;'>A STUDY ON SENTIMENTAL ANALYSIS OF MENTAL ILLNESS, CONNOTATIONS OF TEXTS</h4>", unsafe_allow_html=True)
        st.write("   ")
        st.write("   ")

        st.markdown("<h5 style='text-align: left; font-size:20px;'>Input Your Tweets Here (comma-separated):</h5>", unsafe_allow_html=True)

        user_input = st.text_area("🕊️ Enter Tweets :🕊️")

        if st.button("Predict Sentiment"):
            tweets = user_input.split(',')

            negative_scores = []
            neutral_scores = []
            positive_scores = []

            for i, tweet in enumerate(tweets, start=1):
                tweet = tweet.strip()
                tweet_words = []

                for word in tweet.split(' '):
                    if word.startswith('@') and len(word) > 1:
                        word = '@user'
                    elif word.startswith('http'):
                        word = "http"
                    tweet_words.append(word)

                tweet_proc = " ".join(tweet_words)

                encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
                output = model2(**encoded_tweet)

                scores = output.logits[0].detach().numpy()
                scores = softmax(scores)

                negative_scores.append(scores[0])
                neutral_scores.append(scores[1])
                positive_scores.append(scores[2])

            st.subheader("Sentiment Analysis Results:")
            for i, tweet in enumerate(tweets, start=1):
                st.write(f"**Tweet {i}:** '{tweet}'")
                st.write("Sentiment Scores:")
                st.write(
                    f"<div style='text-align: center; font-size: 28px;'><u>- Negative 😞: {negative_scores[i - 1] * 100:.2f}%</u></div>",
                    unsafe_allow_html=True)
                st.write(
                    f"<div style='text-align: center; font-size: 28px;'><u>- Neutral 😐: {neutral_scores[i - 1] * 100:.2f}%</u></div>",
                    unsafe_allow_html=True)
                st.write(
                    f"<div style='text-align: center; font-size: 28px;'><u>- Positive 😀: {positive_scores[i - 1] * 100:.2f}%</u></div>",
                    unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#f9f9f9')
            x = np.arange(len(tweets))
            colors = ['#ff4c4c', '#cccccc', '#4caf50']
            bar_width = 0.25

            bars1 = ax.bar(x, negative_scores, width=bar_width, label='Negative 😞', color=colors[0])
            bars2 = ax.bar(x + bar_width, neutral_scores, width=bar_width, label='Neutral 😐', color=colors[1])
            bars3 = ax.bar(x + 2 * bar_width, positive_scores, width=bar_width, label='Positive 😀', color=colors[2])

            ax.set_xlabel('Tweets', fontsize=14)
            ax.set_ylabel('Sentiment Scores', fontsize=14)
            ax.set_title('Sentiment Analysis Results', fontsize=16)
            ax.set_xticks(x + bar_width)
            ax.set_xticklabels(tweets, rotation=60, ha="right", fontsize=12)
            ax.legend(fontsize=12)

            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            st.pyplot(fig)

        if st.button("Clear"):
            user_input = ""  # Clear user input
            tweets = []  # Clear stored tweets
            negative_scores = []  # Clear sentiment scores
            neutral_scores = []
            positive_scores = []




    elif Analyzer_selection == "Exploartory Data Analysis":

        st.markdown("<h1 style='text-align: center; color : #FF7074; text-decoration: underline;'>The EDA App</h1>",
                    unsafe_allow_html=True)
        st.write("  ")
        st.markdown(
            "<h2 style='text-align: center; color: #FFFFFF;'>This is the Exploratory Data Analysis App created using the pandas-profiling libraries.</h2>",
            unsafe_allow_html=True)
        with open('caution.txt', 'r') as file:
            caution = file.read()
            st.write(caution)
            st.write("  ")
        with open('EDA_ABT.txt', 'r') as file:
            EDA_ABT = file.read()
            st.write(EDA_ABT)
            st.write("  ")


        st.write("  ")
        st.write("  ")


        st.write(
            "Upload a CSV file to generate an **Exploratory Data Analysis (EDA)** report using the **pandas-profiling** library.")

        st.header("1. Upload Your CSV Data")
        uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

        st.markdown(
            "[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_file'] = uploaded_file.name

            pr = ProfileReport(df, explorative=True)
            st.header("**Input DataFrame**")
            st.write(df)
            st.write("---")
            st.header("**Pandas Profiling Report**")
            st_profile_report(pr)

        else:
            st.info("Awaiting a CSV file to be uploaded.")
            if st.button("Press to Use Example Dataset"):
                df = pd.DataFrame(
                    np.random.rand(100, 5),
                    columns=["1", "2", "3", "4", "5"]
                )
                pr = ProfileReport(df, explorative=True)
                st.header("**Input DataFrame**")
                st.write(df)
                st.write("---")
                st.header("**Pandas Profiling Report**")
                st_profile_report(pr)





elif selected_section == "Algorithm Master":
    if Algorithm_selection == "AutoML":
        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>The Machine Learning Algorithm Comparison App</h1>",
                    unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>In this implementation, the [lazypredict] library is used for building several machine learning models at once.</h3>", unsafe_allow_html=True)
        with open('automl_abt.txt', 'r') as file:
            info = file.read()
            st.write(info)

        def build_model(df):
            df = df.loc[:100]
            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]

            st.markdown('**1.2. Dataset dimension**')
            st.write('X')
            st.info(X.shape)
            st.write('Y')
            st.info(Y.shape)

            st.markdown('**1.3. Variable details**:')
            st.write('X variable (first 20 are shown)')
            st.info(list(X.columns[:20]))
            st.write('Y variable')
            st.info(Y.name)

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=seed_number)
            reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
            models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
            models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

            st.subheader('2. Table of Model Performance')

            st.markdown("<h5 style='text-align: left; text-decoration: underline;'>Training set</h5>",
                        unsafe_allow_html=True)

            st.write(predictions_train)
            st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)

            st.markdown("<h5 style='text-align: left; text-decoration: underline;'>Testing set</h5>",
                        unsafe_allow_html=True)
            st.write(predictions_test)
            st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)

            st.subheader('3. Plot of Model Performance (Test set)')

            sns.set_theme(style="whitegrid")

            predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
            plt.figure(figsize=(12, 8))
            ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test, palette='viridis')
            ax1.set_title('R-Squared Values of Models', fontsize=16)
            ax1.set_xlabel('R-Squared', fontsize=14)
            ax1.set_ylabel('Models', fontsize=14)
            ax1.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(plt)

            predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
            plt.figure(figsize=(10, 6))
            ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test, palette='viridis')
            ax2.set_title('RMSE Values of Models', fontsize=16)
            ax2.set_xlabel('Models', fontsize=9)
            ax2.set_ylabel('RMSE', fontsize=14)
            plt.xticks(rotation=90)
            st.pyplot(plt)

            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(10, 6))
            ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test, palette='viridis')
            ax3.set_title('Calculation Time of Models', fontsize=16)
            ax3.set_xlabel('Models', fontsize=14)
            ax3.set_ylabel('Time Taken (seconds)', fontsize=14)
            plt.xticks(rotation=90)
            st.pyplot(plt)


        def filedownload(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
            return href


        st.header('1. Upload your CSV data')
        (uploaded_file) = st.file_uploader("Upload your input CSV file", type=["csv"])
        st.markdown("""
        [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
        """)

        st.header('2. Set Parameters')
        split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.slider('Set the random seed number', 1, 100, 42, 1)

        st.subheader('1. Dataset')

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)
            build_model(df)
        else:
            st.info('Awaiting for CSV file to be uploaded.')
            if st.button('Press to use Example Dataset'):
                # Diabetes dataset
                diabetes = load_diabetes()
                X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                Y = pd.Series(diabetes.target, name='response')
                df = pd.concat([X, Y], axis=1)

                st.markdown('The Diabetes dataset is used as the example.')
                st.write(df.head(5))

                build_model(df)

    elif Algorithm_selection == "No Code Machine Learning Trainer":

        st.markdown("<h1 style='text-align: center; text-decoration: underline;color : #FF7074;'>🤖 No Code ML Model Training 🤖</h1>",
                    unsafe_allow_html=True)
        with open('caution.txt', 'r') as file:
            caution = file.read()
            st.write("")
            st.write(caution)

        with open('no_code_trainer.txt', 'r') as file:
            info =file.read()
            st.write("")
            st.write(info)

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.write("### Preview of Uploaded Dataset")
            st.dataframe(df.head())

            col1, col2, col3, col4 = st.columns(4)

            scaler_type_list = ["standard", "minmax"]

            # Dictionary made with gpt
            model_dictionary = {
                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBoost Classifier": XGBClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Gaussian Naive Bayes": GaussianNB(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Ridge Classifier": RidgeClassifier(),
                "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                "Extra Trees Classifier": ExtraTreesClassifier(),
                "Multi-Layer Perceptron Classifier": MLPClassifier(),
                "Bagging Classifier": BaggingClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=0)
            }

            with col1:
                target_column = st.selectbox("Select the Target Column", list(df.columns))
            with col2:
                scaler_type = st.selectbox("Select a scaler", scaler_type_list)
            with col3:
                selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
            with col4:
                model_name = st.text_input("Model name")

            if st.button("Train the Model"):
                try:
                    X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                    st.write(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

                    model_to_be_trained = model_dictionary[selected_model]
                    model_to_be_trained.fit(X_train, y_train)

                    # Evaluating using accuracy
                    accuracy = evaluate_model(model_to_be_trained, X_test, y_test)
                    st.success(f"Test Accuracy: {accuracy:.2f}")

                    # sving the model
                    model_buffer = io.BytesIO()
                    pickle.dump(model_to_be_trained, model_buffer)
                    model_buffer.seek(0)

                    if not model_name.strip():
                        model_name = "trained_model"
                    st.download_button(
                        label="Download Trained Model",
                        data=model_buffer,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("Please upload a CSV file to get started.")








































