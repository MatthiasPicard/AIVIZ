import streamlit as st
from utilities.template_helpers import upload_data
from types import NoneType
import extra_streamlit_components as stx

import prince
import plotly.express as px
import pandas as pd

import algos.clustering.kmeans
import algos.clustering.dbscan
import algos.clustering.kproto

import algos.classification.nnclassifier
import algos.classification.logistic
import algos.classification.svmclassifier

import algos.regression.linR
import algos.regression.ridge
import algos.regression.elasticnet

from types import NoneType

def get_data(category, algo_name=None):
    if category in ['Classification','Regression']:
        train = upload_data('Training Data')
        test = upload_data('Testing Data')
        return train, test
    else:
        df = upload_data()
        if type(df) != NoneType:
            return (df,)


def choose_algo(category):
    if category == 'Clustering':
        algo = stx.tab_bar(data=[
            stx.TabBarItemData(id='K-Means',title='K-Means',description='Partitional Clustering Algorithm'),
            stx.TabBarItemData(id='DBSCAN',title='DBSCAN',description='Density Based Clustering Algorithm'),
            stx.TabBarItemData(id='K-Prototype',title='K-Prototype',description='Partitional over Mixed Data')]
        )
        if algo == 'K-Means':
            return algos.clustering.kmeans.process
        if algo == 'DBSCAN':
            return algos.clustering.dbscan.process
        if algo == 'K-Prototype':
            return algos.clustering.kproto.process
    elif category == 'Classification':
        algo = stx.tab_bar(data=[
            stx.TabBarItemData(id='NN',title='Neural Network',description='Multi-Layer Perceptron classifier'),
            stx.TabBarItemData(id='SVM',title='Suport Vector Classifier',
                        description='Classification using Support Vector Machines'),
            stx.TabBarItemData(id='logR',title='Logistic Regression',description='Logistic Regression Classifier')]
        )
        if algo == 'NN':
            return algos.classification.nnclassifier.process
        if algo == 'SVM':
            return algos.classification.svmclassifier.process
        if algo == 'logR':
            return algos.classification.logistic.process
    elif category == 'Regression':
        algo = stx.tab_bar(data=[
            stx.TabBarItemData(id='linR',title='Linear Regression',description='Linear Regression'),
            stx.TabBarItemData(id='ridge',title='Ridge',
                        description='Ridge Regression'),
            stx.TabBarItemData(id='elastic',title='Elastic Net Regression',description='Elastic Net Regression')]
        )
        if algo == 'linR':
            return algos.regression.linR.process
        if algo == 'ridge':
            return algos.regression.ridge.process
        if algo == 'elastic':
            return algos.regression.elasticnet.process


def get_plot(df, title):

    if title == 'Regression':
        return None # Do not plot regression, display its coefficients

    reduce_algo = None
    pca = None

    # Better title for the graph
    viz_thing = 'Clusters'
    if title == 'Classification':
        viz_thing = 'Classes'

    # name of column to represent as color on the graph (target class)
    if type(df) == NoneType:
        return None
    if len(df) == 0:
        return None
    target_class = df.columns[-1]

    if df.shape == (0,0):
        return None

    if 'object' in list(df.dtypes):
        reduce_algo = 'FAMD'
        pca = prince.FAMD(n_components=3)
    else:
        reduce_algo = 'Principal Component Analysis'
        pca = prince.PCA(n_components=3)
    reduced = pca.fit(df.iloc[:,:-1]).row_coordinates(df.iloc[:,:-1])
    reduced.columns = ['X','Y','Z']
    reduced[target_class] = df[target_class].astype(str)
    # Each axe's inertia
    labs = {
        "X" : f"Component 0 - ({round(100*pca.explained_inertia_[0],2)}% inertia)",
        "Y" : f"Component 1 - ({round(100*pca.explained_inertia_[1],2)}% inertia)",
        "Z" : f"Component 2 - ({round(100*pca.explained_inertia_[2],2)}% inertia)",
    }
    tot_inertia = f"{round(100*pca.explained_inertia_.sum(),2)}"
    st.write(f'{reduce_algo} Visualization of {viz_thing} ({tot_inertia}%) :')
    fig = px.scatter_3d(reduced,x='X',y='Y',z='Z',color=target_class,labels=labs)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),showlegend=False,height=300)
    return fig