import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier




# primaryColor="#6eb52f"
# backgroundColor="#f0f0f5"
# secondaryBackgroundColor="#e0e0ef"
# textColor="#262730"
# font="sans serif"




st.title('Iris Flower Prediction App')
st.write("""



This app predicts the Iris flower type!
""")


st.sidebar.header('user Input parameters')

def user_input_features():
    sepal_length=st.sidebar.slider('sepal_length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.3)
    petal_width= st.sidebar.slider('petal_width', 0.1, 2.5, 0.2)
    st.sidebar.markdown("<h1 style='text-align: right; color:gray;font-size:18px'>-by kesavakumar pommu</h1>",
                unsafe_allow_html=True)
    data={' sepal_length': sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])

    return features

df=user_input_features()

st.subheader('user input parameter')
st.write(df)

iris=datasets.load_iris()
X=iris.data
Y=iris.target


clf=RandomForestClassifier()
clf.fit(X,Y)


prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('prediction')
st.write(iris.target_names[prediction])

st.subheader('prediction Probility')
st.write(prediction_proba)
