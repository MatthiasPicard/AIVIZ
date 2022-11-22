import pandas as pd
import streamlit as st


def upload_data(descr='Upload Data'):
    up = st.file_uploader(descr)
    if up:
        df = pd.read_csv(up).dropna()
        return df