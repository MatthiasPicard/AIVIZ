import streamlit as st
from utilities.components import get_data, choose_algo, get_plot
from types import NoneType
import pandas as pd


def get_info(category):
    infos = {
        " --- Choose --- ":'We Provide several different types of algorithms, such as Clustering or Classification',
        "Clustering":'Unsupervised, creates clusters of similars individuals',
        "Classification":'Supervised, assigns individuals to a class usign training data',
        "Others":'Other algorithms, such as linear regression'
    }
    st.info(infos[category])

class Page:
    def __init__(self, title) -> None:
        self.title = title
        self.data = None
        self.algo = None
        self.plot = None
        self.results = None
    
    def render(self):
        st.title(self.title.upper())
        col1, col2 = st.columns([2,5])

        ##### CHOOSE DATA #####
        with col1.container():
            data = get_data(self.title)
            if type(data) == tuple:
                if type(data[0]) != NoneType:
                    st.dataframe(data[0].head(5//len(data)))
            self.data = data
            

        with col2.container():
            ##### CHOSE ALGORITHM #####
            self.algo = choose_algo(self.title)
            if self.algo is not None and self.data is not None:
                self.results = pd.DataFrame(self.algo(self.data))
                self.plot = get_plot(self.results)

            ##### PLOT RESULTS #####
            if self.plot is not None:
                st.plotly_chart(self.plot)
            
        ##### DOWNLOAD RESULTS #####
        if self.results is not None:
            col1.download_button("Download Results",
                            self.results.to_csv(index=False),
                            "results.csv",
                            "text/csv", 
                            key="download-csv")