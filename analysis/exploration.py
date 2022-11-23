import streamlit as st
from utilities.template_helpers import upload_data
import pandas as pd
from types import NoneType
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sys

def render():
    st.title("DATA EXPLORATION")
    col1, col2 = st.columns([2,5])
    df = None
    with col1.container():
        df = upload_data()
        if type(df) is NoneType:
            return
        st.dataframe(df.describe())
    with col2.container():
        pr = ProfileReport(df)
        st_profile_report(pr)