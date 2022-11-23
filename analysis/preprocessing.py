import streamlit as st
from utilities.template_helpers import upload_data
from types import NoneType
import pandas as pd
import numpy as np


def render():
    st.title("PREPROCESSING")
    # dropna
    # fillna
    # select columns
    # scaling

    col1, col2, col3 = st.columns([1,1,1])

    df = None
    with col1.container():
        df = upload_data()
        if type(df) is NoneType:
            return
        if df.shape == (0,0):
            st.write('bof')
            return
        info = pd.DataFrame()
        info['dtypes'] = pd.DataFrame(df.dtypes)
        info['null'] = df.isna().sum()
        st.dataframe(info,use_container_width=True)

    with col2.container():
        ### DROP NA ###
        st.write('\n\n')
        st.markdown('#### Drop Null Values')
        st.write('Drop any row containing null values')
        drop_null = st.button('Drop')
        if drop_null:
            df.dropna(inplace=True)

        ### FILL NA ####
        st.write("\n\n")
        st.markdown('#### Fill Null Values')
        st.write("""Replace null values with mean of the column for numerical variables,
                     and mode for categorical variables""")
        fill_null = st.button('fill')
        if fill_null:
            for col in df.columns:
                val = 0
                if df[col].dtype == 'object':
                    val = df[col].mode(dropna=True)
                else:
                    val = df[col].mean(dropna=True)
                df[col].fillna(val)