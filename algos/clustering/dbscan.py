import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np


def process(data):

    if 'object' in list(data[0].dtypes):
        st.info('This Algorithm can only process numerical data')
        return None

    scaler = StandardScaler()
    df = data[0].copy()

    for c in data[0].columns:
        df[c] = scaler.fit_transform(data[0][[c]])

    max_distance = st.slider("""Maximum distance between two samples for one to be considered
                                as in the neighborhood of the other. :""",0.01,5.0)
    dbscan = DBSCAN(max_distance)
    res = dbscan.fit_predict(df)
    df = data[0]
    df['cluster'] = res
    return df