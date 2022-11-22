import streamlit as st
from utilities.standard_template import Page, get_info
from utilities.land import land_page
import warnings
 
warnings.filterwarnings("ignore")

# PAGE CONFIGURATION, CHANGE NAME AND ICON

st.set_page_config(layout="wide",page_title='AIViz')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
    choice = st.selectbox('Choose Algorithm Category',[
                                                " --- Choose --- ",
                                                "Clustering",
                                                "Classification",
                                                "Regression",
                                                "Others"
                                            ])
    get_info(choice)

if choice != ' --- Choose --- ':
    Page(choice).render()
    
    
else:
    land_page()