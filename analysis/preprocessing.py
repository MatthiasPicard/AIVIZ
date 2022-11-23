import streamlit as st
from utilities.template_helpers import upload_data
from types import NoneType
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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
            return
        info = pd.DataFrame()
        info['dtypes'] = pd.DataFrame(df.dtypes)
        info['null'] = df.isna().sum()

        tab1, tab2 = st.tabs(['Dataframe','Info'])
        with tab1:
            st.dataframe(df, use_container_width=True, height=300)
        with tab2:
            st.dataframe(info,use_container_width=True,height=300)

    with col2.container():
        ### DROP NA ###
        st.write('\n\n')
        st.markdown('#### Drop Null Values')
        st.write('Drop any row containing null values')
        drop_null = st.checkbox('Drop')
        if drop_null:
            df.dropna(inplace=True)

        ### FILL NA ####
        st.write("\n\n")
        st.markdown('#### Fill Null Values')
        st.write("""Replace null values with mean of the column for numerical variables,
                     and mode for categorical variables""")
        fill_null = st.checkbox('Fill')
        if fill_null:
            for col in df.columns:
                val = 0
                if df[col].dtype == 'object':
                    val = df[col].mode()
                else:
                    val = df[col].mean()
                df[col].fillna(val)

        ### SCALING ###
        st.write('\n\n')
        st.markdown("#### Scaling")
        st.write("Standardize numerical features by removing the mean and scaling to unit variance.")
        scale = st.checkbox('Scale')
        if scale:
            numerical_columns = df.select_dtypes('number').columns
            categorical_columns = df.select_dtypes('object').columns
            categorical_indexes = []

            # Scaling
            scaler = StandardScaler()
            for c in categorical_columns:
                categorical_indexes.append(df.columns.get_loc(c))
            # create a copy of our data to be scaled
            df_scale = df.copy()
            # standard scale numerical features
            for c in numerical_columns:
                df_scale[c] = scaler.fit_transform(df[[c]])
            df = df_scale


    with col3.container():
        ### SELECT COLUMNS
        st.write("\n\n")
        st.markdown("#### Choose columns")
        cols = st.multiselect('Select columns to use',options=list(df.columns),default=list(df.columns))
        #select_cols = st.button('Use selected columns')
        #if select_cols:
        df = df[cols]

        st.write("\n\n")
        st.markdown("#### Encode Numerical values")
        enc = st.checkbox('Encode')
        if enc:
            df.loc[:,df.dtypes == 'object']=df.loc[:,df.dtypes == 'object'].apply(
            lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

        st.write('\n\n')
        st.markdown("#### Download Preprocessed data")
        st.download_button("Download Results",
                            df.to_csv(index=False),
                            "preprocessed.csv",
                            "text/csv", 
                            key="download-csv")
        #st.dataframe(df)




#def res_session():
#    st.session_state['drop_na'] = False
#    st.session_state['fill_na'] = False
#    st.session_state['scale'] = False
#    st.session_state['']