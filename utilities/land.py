import streamlit as st



def land_page():
    _,center,_ = st.columns([2,3,2])
    center.markdown("<h1 style='text-align: center;'>AIViz</h1>", unsafe_allow_html=True)
    center.write("""Machine Learning. For everyone. Now. AIViz is a platform built to let everyone perform Machine
    Learning easily on their own data.""")

    center.image('carott.png')

    center.markdown("<h3 style='text-align: center;'>Use your own data</h3>", unsafe_allow_html=True)

    center.write("You can use your own data with AIViz. All you need is clicking a button. ")

    center.markdown("<h3 style='text-align: center;'>Understand your Data</h3>", unsafe_allow_html=True)

    center.write("""AIViz provides a Data Exploration tool, that lets you explore all your variables. You can
                easily visualize and understand univariate and bivariate behaviors of your data. """)

    center.markdown("<h3 style='text-align: center;'>Preprocessing</h3>", unsafe_allow_html=True)

    center.write("""You can prepare your data for Machine Learning in just a few clicks. You can decide how
                    to handle missing values, choose which columns to use, scale your data...""")

    center.markdown("<h3 style='text-align: center;'>Machine Learning</h3>", unsafe_allow_html=True)

    st.latex("""The \ smartest \ carott \ of \ the \ World \\newline \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
        \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
        \ \ \ \ \ \ \ \ \ \ \ \ \   - \ us.""")   

    center.write("""The core of AIViz is Machine Learning. Now that you have uploaded and preprocessed
                your data, you can perform Artificial Intelligence algorithms on it. We provide several
                different algorithms, for Clustering, Classification or Regression.""")
