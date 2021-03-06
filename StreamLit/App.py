import streamlit as st
import numpy as np
from resizeimage import resizeimage
import tensorflow as tf
import os
from PIL import Image

# model = tf.keras.models.load_model('effmodel.hdf5')
eff_model = tf.keras.models.load_model(os.path.join("StreamLit/Models/","effmodel.hdf5"))
mobile_model = tf.keras.models.load_model(os.path.join("StreamLit/Models/","MobileNet.hdf5"))


label={ 'alpine sea holly': 0,
        'anthurium': 1,
        'artichoke': 2,
        'azalea': 3,
        'balloon flower': 4,
        'barberton daisy': 5,
        'bee balm': 6,
        'bird of paradise': 7,
        'bishop of llandaff': 8,
        'black-eyed susan': 9,
        'blackberry lily': 10,
        'blanket flower': 11,
        'bolero deep blue': 12,
        'bougainvillea': 13,
        'bromelia': 14,
        'buttercup': 15,
        'californian poppy': 16,
        'camellia': 17,
        'canna lily': 18,
        'canterbury bells': 19,
        'cape flower': 20,
        'carnation': 21,
        'cautleya spicata': 22,
        'clematis': 23,
        "colt's foot": 24,
        'columbine': 25,
        'common dandelion': 26,
        'common tulip': 27,
        'corn poppy': 28,
        'cosmos': 29,
        'cyclamen': 30,
        'daffodil': 31,
        'daisy': 32,
        'desert-rose': 33,
        'fire lily': 34,
        'foxglove': 35,
        'frangipani': 36,
        'fritillary': 37,
        'garden phlox': 38,
        'gaura': 39,
        'gazania': 40,
        'geranium': 41,
        'giant white arum lily': 42,
        'globe thistle': 43,
        'globe-flower': 44,
        'grape hyacinth': 45,
        'great masterwort': 46,
        'hard-leaved pocket orchid': 47,
        'hibiscus': 48,
        'hippeastrum': 49,
        'iris': 50,
        'japanese anemone': 51,
        'king protea': 52,
        'lenten rose': 53,
        'lilac hibiscus': 54,
        'lotus': 55,
        'love in the mist': 56,
        'magnolia': 57,
        'mallow': 58,
        'marigold': 59,
        'mexican petunia': 60,
        'monkshood': 61,
        'moon orchid': 62,
        'morning glory': 63,
        'orange dahlia': 64,
        'osteospermum': 65,
        'passion flower': 66,
        'peruvian lily': 67,
        'petunia': 68,
        'pincushion flower': 69,
        'pink primrose': 70,
        'pink quill': 71,
        'pink-yellow dahlia': 72,
        'poinsettia': 73,
        'primula': 74,
        'prince of wales feathers': 75,
        'purple coneflower': 76,
        'red ginger': 77,
        'rose': 78,
        'ruby-lipped cattleya': 79,
        'siam tulip': 80,
        'silverbush': 81,
        'snapdragon': 82,
        'spear thistle': 83,
        'spring crocus': 84,
        'stemless gentian': 85,
        'sunflower': 86,
        'sweet pea': 87,
        'sweet william': 88,
        'sword lily': 89,
        'thorn apple': 90,
        'tiger lily': 91,
        'toad lily': 92,
        'tree mallow': 93,
        'tree poppy': 94,
        'trumpet creeper': 95,
        'wallflower': 96,
        'water lily': 97,
        'watercress': 98,
        'wild geranium': 99,
        'wild pansy': 100,
        'wild rose': 101,
        'windflower': 102,
        'yellow iris': 103}

key_list = list(label.keys())
val_list = list(label.values())
    
def validate_set(img):
    image = np.array(img) 
#     st.write(image.shape)
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
#         st.write("eff doneee")
    if choose_model == 'MobileNet':
        model = mobile_model
#         st.write("mobile doneee")
   
    
    if(file):
        st.image(file)
        image = Image.open(file)
        image = resizeimage.resize_cover(image, [224, 224])
        img_val = validate_set(image)
        y_pred = model.predict(img_val)
        Y_pred_classes = np.argmax(y_pred,axis=1)
#         st.success(Y_pred_classes)
        position = val_list.index(Y_pred_classes)
        st.success(key_list[position])
        
        
      
    
    
    
    
    

    
if __name__ == '__main__':
    main()  
    
    

