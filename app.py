# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:14:39 2020

@author: DELL
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image
filename = 'covid-text-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('cv-transform.pkl','rb'))
def predict(text):
    text=[text]
    text=cv.transform(text).toarray()
    prediction=classifier.predict(text)
    if prediction==0:
        prediction="This is normal text"
    else:
        prediction="This text regrading covid-19"
    return prediction



def main():
    st.title("Text-classifier")
    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">covid-text-classifier ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text = st.text_input("Text"," ")
    result=""
    if st.button("Predict"):
        result=predict(text)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()