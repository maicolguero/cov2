# importar libreria
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
from skimage.transform import resize

tf.function(experimental_relax_shapes=False)

# Modelos entrenados
MODELO = 'covid.h5'

# Dimensiones de las imagenes de entrada    
width_shape = 250
height_shape = 250

# Clases
names = ['covid','nocovid']


def model_prediction(img, model):

    img_resize = resize(img, (width_shape, height_shape))
    x=preprocess_input(img_resize*255)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds

st.title(" AYUDA PARA EL DIAGNOSTICO DE COVID")


def main():
    
    

    # Se carga el modelo

    model = load_model(MODELO)
        
    predictS=""
    st.write("programa de reconocimiento de tomografías axiales computarizadas del SARS-CoV-2")
    img_file_buffer = st.file_uploader("Carge una imagen ", type=["png", "jpg", "jpeg"])
        
     # El usuario carga una imagen
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))    
        st.image(image, caption="Imagen", use_column_width=False)
        
        # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
        predictS = model_prediction(image, model)
        st.success('EL PACIENTE ES: {}'.format(names[np.argmax(predictS)]))
    
if __name__ == '__main__':
    main()