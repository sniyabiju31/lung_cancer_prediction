import streamlit as st
import pickle
from PIL import Image
def main():
    st.title(":red[LUNG CANCER PREDICTION]")
    img=Image.open("lungcancerlu.jpeg")
    st.image(img,width=800)
    age=st.text_input("Enter age of patient","age here")
    smoke=st.text_input("Average number of ciggarates smoke in a day","")
    areaq=st.text_input("Quality of living area","")
    alco=st.text_input("Average alcoholic consumption per day","")
    features=[age,smoke,areaq,alco]
    minmax=pickle.load(open('scalar.sav','rb'))
    model=pickle.load(open('knn_model.sav','rb'))
    pred=st.button("PREDICT")
    if pred:
        result=model.predict(minmax.transform([features]))
        if result==0:
            st.write("# Person will not suffer lung cancer")
        else:
            st.write("# Person will suffer lung cancer")
main()