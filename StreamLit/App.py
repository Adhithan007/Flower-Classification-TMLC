import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model('StreamLit/effmodel2.hdf5')

def validate_set(img):

    X_valid = []

        #image = ImageOps.grayscale(image)
        
    image = np.array(img)
    image_data_as_arr = np.asarray(image)
        
    X_valid.append(image_data_as_arr)
    X_valid = np.asarray(X_valid)   
    X_valid = tf.expand_dims(X_valid, axis=-1)
    return X_valid

# file=False
def main():
  
    st.write("""
             # Flower Image Classification
             """
             )
    st.write("This is a simple image classification web app to predict the name of the Flower")
    
    file = st.file_uploader("Please upload an image file {.jpg| .jpeg}", type=["jpg","jpeg"])
    choose_model = st.selectbox('Select a trained model:', ('EfficientNet'))
    if choose_model == 'EfficientNet':
        model = model
   
    
    if(file):
        img_val = validate_set(file)
        y_pred = model1.predict(img_val)
        Y_pred_classes = np.argmax(y_pred,axis=1)
        st.success(Y_pred_classes)
        
        
      
    
    
    
    
    
    # Sidebar Column
    st.sidebar.title('Sidebar Widgets')
    #radio button 
    rating = st.sidebar.radio('Are You Happy with the Example',('Yes','No','Not Sure'))
    if rating == 'Yes':
        st.sidebar.success('Thank You for Selecting Yes')
    elif rating =='No':
        st.sidebar.info('Thank You for Selecting No')
    elif rating =='Not Sure':
        st.sidebar.info('Thank You for Selecting Not sure')
    #selectbox
    rating = st.sidebar.selectbox("How much would you rate this App? ",
                         ['5 Stars', '4 Stars', '3 Stars','2 Stars','1 Star'])
    
if __name__ == '__main__':
    main()  
    
    
