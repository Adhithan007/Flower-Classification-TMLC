import streamlit as st
import numpy as np
from resizeimage import resizeimage
import tensorflow as tf
import os

# model = tf.keras.models.load_model('effmodel.hdf5')
eff_model = tf.keras.models.load_model(os.path.join("StreamLit/Models/","effmodel.hdf5"))


def validate_set(img):
    image = np.array(img) 
    st.write(image.shape)
    X_valid = image.reshape(-1, 224, 224, 3)
    return X_valid



# file=False
def main():
    
    

        
    
    st.write("""
             # Flower Image Classification
             """
             )
    st.write("This is a simple image classification web app to predict the name of the Flower")



    file = st.file_uploader("Please upload an image file {.jpg| .jpeg}", type=["jpg","jpeg"])
 
    choose_model = st.selectbox('Select a trained model:', ('MobileNet','EfficientNet'))
    if choose_model == 'EfficientNet':
        model = eff_model
        st.write("doneee")
   
    
    if(file):
        st.image(file)
        image = resizeimage.resize_cover(image, [224, 224])
        img_val = validate_set(image)
        y_pred = model1.predict(img_val)
        Y_pred_classes = np.argmax(y_pred,axis=1)
        st.success(Y_pred_classes)
        
        
      
    
    
    
    
    

    
if __name__ == '__main__':
    main()  
    
    

