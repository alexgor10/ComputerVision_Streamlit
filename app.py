import streamlit as st
import requests  
import json  

from PIL import Image
from io import BytesIO

from transformers import ViTFeatureExtractor, ViTForImageClassification

url =  "https://stablediffusionapi.com/api/v4/dreambooth"  

headers =  {  
    'Content-Type':  'application/json'  
}

def main_page():

    # Configuración inicial de la página de Streamlit
    st.set_page_config(layout='wide')

    # Título de la aplicación
    st.title('Simple Streamlit App')
    
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            # Crear un campo de entrada de texto
            input_text = st.text_input('Ingrese algún texto')

            payload = json.dumps({  
                "key":  "your_api_key",  
                "model_id":  "juggernaut-xl-v5",  
                "prompt":  input_text,  
                "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
                "width":  "512",  
                "height":  "512",  
                "samples":  "1",  
                "num_inference_steps":  "30",  
                "safety_checker":  "no",  
                "enhance_prompt":  "yes",  
                "seed":  None,  
                "guidance_scale":  7.5,  
                "multi_lingual":  "no",  
                "panorama":  "no",  
                "self_attention":  "no",  
                "upscale":  "no",  
                "embeddings":  "embeddings_model_id",  
                "lora":  "lora_model_id",  
                "webhook":  None,  
                "track_id":  None  
                }) 

            
            # Crear un botón
            if st.button('Mostrar Imagen'):
                try: 
                    response = requests.request("POST", url, headers=headers, data=payload)
                    response.raise_for_status()

                    data = response.json()

                    if 'status' in data and data['status'] == 'error':
                        error = data['message']
                        st.markdown(f'<div style="color: white;">{error}</div>', unsafe_allow_html=True)
                        print(error)
                    else:
                        image = data['future_links'][0]

                        if image:
                            st.image(image, caption="Imagen desde URL", use_column_width=True)
                        else:
                            st.warning("Error al cargar la imagen")
                    
                except requests.exceptions.HTTPError as errh:
                    print(f"Error HTTP: {errh}")
                except requests.exceptions.ConnectionError as errc:
                    print(f"Error de conexión: {errc}")
                except requests.exceptions.Timeout as errt:
                    print(f"Error de tiempo de espera: {errt}")
                except requests.exceptions.RequestException as err:
                    print(f"Error general de solicitud: {err}")

        
        with col2:
            uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Mostrar la imagen cargada
                st.image(uploaded_file, caption="Imagen Cargada")

                # Procesar la imagen
                im = Image.open(uploaded_file)
                # Init model, transforms
                model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
                transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

                # Transform our image and pass it through the model
                inputs = transforms(im, return_tensors='pt')
                output = model(**inputs)

                # Predicted Class probabilities
                proba = output.logits.softmax(1)

                # Predicted Classes
                preds = proba.argmax(1).item()

                if preds == 0:
                    st.markdown('<div style="color: white;">Edad entre 0 - 2 años</div>', unsafe_allow_html=True)
                elif preds == 1:
                    st.markdown('<div style="color: white;">Edad entre 3 - 9 años</div>', unsafe_allow_html=True)
                elif preds == 2:
                    st.markdown('<div style="color: white;">Edad entre 10 - 19 años</div>', unsafe_allow_html=True)
                elif preds == 3:
                    st.markdown('<div style="color: white;">Edad entre 20 - 29 años</div>', unsafe_allow_html=True)
                elif preds == 4:
                    st.markdown('<div style="color: white;">Edad entre 30 - 39 años</div>', unsafe_allow_html=True)
                elif preds == 5:
                    st.markdown('<div style="color: white;">Edad entre 40 - 49 años</div>', unsafe_allow_html=True)
                elif preds == 6:
                    st.markdown('<div style="color: white;">Edad entre 50 - 59 años</div>', unsafe_allow_html=True)
                elif preds == 7:
                    st.markdown('<div style="color: white;">Edad entre 60 - 69 años</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color: white;">Mas de 70 años</div>', unsafe_allow_html=True)


# Punto de entrada principal de la aplicación Streamlit
if __name__ == "__main__":
    main_page()