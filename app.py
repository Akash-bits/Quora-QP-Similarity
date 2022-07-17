import streamlit as st
import pickle
import helper
import numpy as np

# pickle_off = open("model.pkl","rb")
# rf = pickle.load(pickle_off)

with open('model.pkl','rb') as f:
  rf=pickle.load(f)

st.header('Quora Question Pair Similarity')

q1 = st.text_input('Enter 1st question')
q2 = st.text_input('Enter 2nd question')
#print(q1)
#print(q2)

if st.button('Predict Similarity'):
    query = np.array(helper.computeQueryPoint(q1,q2)).reshape(1,-1)
    #print(query)
    result1 = rf.predict(query)
    print(result1)
    result = result1[0]
    #print(result)

    if (result==1):
        st.header('Similar Questions')
    else:
        st.header('Not Similar Questions')
