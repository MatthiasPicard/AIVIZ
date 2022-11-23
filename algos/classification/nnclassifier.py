import streamlit as st
from sklearn.neural_network import MLPClassifier
import pandas as pd
from types import NoneType

def process(data):
    if type(data[0]) == NoneType or type(data[1]) == NoneType: # if either training or testing dataset is still missing
        st.info('Please Upload Data')
        return None
    x_train = data[0].iloc[:,:-1]
    y_train = data[0].iloc[:,-1]
    #st.write(x_train.shape)
    x_test = data[1].iloc[:,:x_train.shape[1]]
    #st.dataframe(data[1])
    #st.write(x_test.shape)
    
    if len(x_train.columns) != len(x_test.columns):
        st.info('Training and testing datasets have different column number, cannot perform classification.')
        return None

    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
    pred = clf.predict(x_test)
    #x_test[data[0].columns[-1]] = pred
    x_test[data[0].columns[-1]] = pred
    #st.dataframe(x_test)
    return x_test