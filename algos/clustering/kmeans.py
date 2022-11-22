import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.cluster import KMeans



def process(data):

    if 'object' in list(data[0].dtypes):
        st.info('This Algorithm can only process numerical data')
        return None

    scaler = StandardScaler()
    df = data[0].copy()

    for c in data[0].columns:
        df[c] = scaler.fit_transform(data[0][[c]])
    k = st.slider('Number of Clusters :',2,9)
    kmeans = KMeans(k)
    res = kmeans.fit_predict(df)
    df = data[0]
    df['cluster'] = res
    return df