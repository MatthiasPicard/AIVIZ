from sklearn.linear_model import LinearRegression
import streamlit as st
from types import NoneType

def process(data):
    if type(data[0]) == NoneType or type(data[1]) == NoneType: # if either training or testing dataset is still missing
        st.info('Please Upload Data')
        return None
    if len(data) == 0:
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
    if 'object' in list(data[0].dtypes) or 'object' in list(data[1].dtypes):
        st.info('Please Upload Numerica Data.')
        return None

    reg = LinearRegression().fit(x_train, y_train)


    cols = x_train.columns
    #st.write(list(zip(reg.coef_,cols)))
    st.latex(f"  {x_train.columns[-1]} =   ")
    coeffs = ['{:.4f}'.format(float(c)) for c in reg.coef_]

    eq = ' + '.join([str(col) +' × '+ (alpha) for col,alpha in zip(coeffs,cols)])
    st.markdown(f" $$ {reg.intercept_} {eq} $$")

    st.latex(f" R² = {reg.score(x_train, y_train)} ")

    pred = reg.predict(x_test)
    x_test[data[0].columns[-1]] = pred
    return x_test