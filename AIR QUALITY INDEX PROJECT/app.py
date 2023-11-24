import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

#Load up the model
model = joblib.load(open('C:\\Users\\TeeFaith\\Desktop\\ML PROJECTS\\AIR QUALITY INDEX PROJECT\\AQI_MODEL', 'rb'))


def AQI_pred(PM, SO, NO, PMT, NH, O, CO, BEN):
    pred_arr = np.asarray([PM, SO, NO, PMT, NH, O, CO, BEN])
    pred_reshape = pred_arr.reshape(1,-1)
    #pred_int = pred_reshape.astype(int)
    scaler = StandardScaler()
    pred_scaled = scaler.fit_transform(pred_reshape)

    model_prediction = model.predict(pred_scaled)
    return model_prediction

def run():
    st.title('AIR QUALITY INDEX FORCAST')

    PM = st.text_input('PM10 value')
    SO = st.text_input('SO2 value')
    NO = st.text_input('NOx value')
    PMT = st.text_input('PM 2.5 value')
    NH = st.text_input('Ammonia Value')
    O = st.text_input('Ozone value')
    CO = st.text_input('CO value')
    BEN = st.text_input('Benzene Value')

    prediction = ''

    if st.button('predict'):
        prediction= AQI_pred(PM, SO, NO, PMT, NH, O, CO, BEN)

    st.success('The model predicts:{}'.format(prediction))

if __name__ == '__main__':
    run()