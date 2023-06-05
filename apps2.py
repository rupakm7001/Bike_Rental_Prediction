# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Load the model
model = pickle.load(open('rf.pkl', 'rb'))

# Load the dataframe
df = pd.read_csv("C:\\Users\\rajpu\\project_3\\df.csv")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('pic.jpg')

selected=st.sidebar.radio("main page",['deployment','Visualisation'])
selected

if selected == 'deployment':
    image = Image.open("C:\\Users\\rajpu\\project_3\\pic1.jpg")
    st.image(image, caption = 'Service Logo')
    
    def predict(mnth, hr, weekday, temp, hum, windspeed, casual, season_fall, season_springer, season_summer, season_winter, holiday_No, holiday_Yes, workingday_No_work, workingday_Working_Day, weathersit_Clear, weathersit_HeavyRain, weathersit_LightSnow, weathersit_Mist):

        input_data = np.array([[mnth, hr, weekday, temp, hum, windspeed, casual, season_fall, season_springer, season_summer, season_winter, holiday_No, holiday_Yes, workingday_No_work, workingday_Working_Day, weathersit_Clear, weathersit_HeavyRain, weathersit_LightSnow, weathersit_Mist]])
         
    # Create an input array
         

    # Make the prediction using the trained model
        prediction = model.predict(input_data)

    # Return the prediction
        return prediction[0]

  # Price Calculator page
    def main():
        st.title("Bike Sharing Prediction App")
        st.write("Enter the values for the features to get a prediction")

        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        weekday_names = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}

        mnth = st.selectbox("Month", options=list(month_names.values()), index=0)
        hr = st.number_input("Hour", min_value=0, max_value=23, value=12)
        weekday = st.selectbox("Weekday", options=list(weekday_names.values()), index=0)
        temp = st.number_input("Temperature (in Celsius)", value=25.0)
        hum = st.number_input("Humidity (in %)", min_value=0, max_value=100, value=50)
        windspeed = st.number_input("Wind Speed (in km/h)", value=25.0)
        casual = st.number_input("Casual", value=0)
        "select Season"
        season_fall = st.checkbox("Fall")
        season_springer = st.checkbox("Spring")
        season_summer = st.checkbox("Summer")
        season_winter = st.checkbox("Winter")
        "Type of day"
        holiday_No = st.checkbox("No Holiday")
        holiday_Yes = st.checkbox("Holiday")
        "Type of working Day"
        workingday_No_work = st.checkbox("No Working Day")
        workingday_Working_Day = st.checkbox("Working Day")
        "Weather Situation"
        weathersit_Clear = st.checkbox("Clear")
        weathersit_HeavyRain = st.checkbox("Heavy Rain")
        weathersit_LightSnow = st.checkbox("Light Snow")
        weathersit_Mist = st.checkbox("Mist")

        # Map the selected values back to numerical values
        mnth = list(month_names.keys())[list(month_names.values()).index(mnth)]
        weekday = list(weekday_names.keys())[list(weekday_names.values()).index(weekday)]

        # Make the prediction using the predict() function
        prediction = predict(mnth, hr, weekday, temp, hum, windspeed, casual, season_fall, season_springer, season_summer, season_winter, holiday_No, holiday_Yes, workingday_No_work, workingday_Working_Day, weathersit_Clear, weathersit_HeavyRain, weathersit_LightSnow, weathersit_Mist)

        # Display the prediction
        st.write("The predicted number of bikes rented is:", round(prediction))

        # Visualize the data with the best fitted line
        
        
  
    
    

    if __name__ == '__main__':
        main()





elif selected=='Visualisation':
    image1 = Image.open("C:\\Users\\rajpu\\project_3\\pic2.jpg")
    st.image(image1, caption = 'Institute Logo')
    
    st.title("Visualisation of WebApp Bike Rental Sharing")
    st.write("Line Chart shows the trends")
    
    cnt=df["cnt"].unique().tolist()
    hr=df["hr"].unique().tolist()
    weekday=df["weekday"].unique().tolist()
    
    windspeed=df["windspeed"].unique().tolist()
    
    df1=pd.DataFrame(np.random.randn(20, 2),
    columns=['cnt','hr'])
    st.line_chart(df1)
    
    df2=pd.DataFrame(np.random.randn(20, 2),
    columns=['cnt','weekday'])
    st.line_chart(df2)
    
    df3=pd.DataFrame(np.random.randn(20, 2),
    columns=['cnt','windspeed'])
    st.line_chart(df3)
    
    
    df3=pd.DataFrame(np.random.randn(20, 2),
    columns=['cnt','windspeed'])
    st.area_chart(df3)
   
    