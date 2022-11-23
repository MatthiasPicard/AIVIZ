from sklearn.linear_model import Ridge
import streamlit as st
from types import NoneType

def process(data):
    if type(data[0]) == NoneType or type(data[1]) == NoneType: # if either training or testing dataset is still missing
        st.info('Please Upload Data')
        return None
    if len(data) == 0:
        st.info('Please Upload Data')
        return None
    if 'object' in list(data[0].dtypes) or 'object' in list(data[1].dtypes):
        st.info('Please Upload Numerica Data.')
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

    clf = Ridge(alpha=1.0).fit(x_train, y_train)
    pred = clf.predict(x_test)
    #st.write(clf.coef_)

    cols = x_train.columns
    st.latex(f"  {data[0].columns[-1]} =   ")
    coeffs = ['{:.4f}'.format(float(c)) for c in clf.coef_]
    eq = ' + '.join([str(col) +' × '+ (alpha) for col,alpha in zip(coeffs,cols)])
    st.markdown(f" $$ {clf.intercept_} + {eq} $$")
    st.latex(f" R² = {clf.score(x_train, y_train)} ")   
    x_test[data[0].columns[-1]] = pred
    return x_test