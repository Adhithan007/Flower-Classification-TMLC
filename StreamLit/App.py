import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

model = load_model('effmodel.hdf5')

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
   
    
#     if(file):
#         img_val = validate_set(file)
#         y_pred = model1.predict(img_val)
#         Y_pred_classes = np.argmax(y_pred,axis=1)
#         st.success(Y_pred_classes)
        
        
      
    
    
    
    
    

    
if __name__ == '__main__':
    main()  
    
    
