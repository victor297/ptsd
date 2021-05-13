import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import os
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Model, model_from_json
import time
from bokeh.models.widgets import Div

# class
class FaEmoModel(object):
    
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        #set_session(session)
        self.preds = self.loaded_model.predict(img)
        return FaEmoModel.EMOTIONS_LIST[np.argmax(self.preds)]

#importing the cnn model+using the CascadeClassifier to use features at once to check if a window is not a face region
st.set_option('deprecation.showfileUploaderEncoding', False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FaEmoModel("model.json", "best_accuracy_weights.h5")
#model = model_from_json(open("model.json", "r").read())
#model.load_weights('best_accuracy_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX 

#facial expressions detecting function
def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:

			fc = gray[y:y+h, x:x+w]
			roi = cv2.resize(fc, (48, 48))
			pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
			cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
			return img,faces,pred 

#the main function
def main():

    """Face Expression Detection App"""
    #setting the app title & sidebar

    activities = ["Home", "Recognize My Facial Expressions" , "See Model Performance", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Home':
        html_temp = """ 
        <marquee behavior="scroll" direction="left" width="100%;">
        <h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">How are you feeling today? </h2>
        </marquee><br>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        st.subheader("**Video Demo: **")
        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        st.markdown("____")
        st.markdown("**Working on bringing it to you...**")
        st.markdown("**Will be uploaded soon!**")

    if choice == 'See Model Performance':
        st.title("Web Application For Facial Expression Recognition!")
        #html_choice = """ 
        #<marquee behavior="scroll" direction="left" width="100%;">
        #<h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        #</marquee><br>
        #"""
        #st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
        st.markdown("____")
        #st.subheader("\nWorking on bringing it to you...")
        #st.subheader("Will be back with it soon!")
        actions = ["Train and Test Accuracies", "Evaluation of the Model" , "Confusion Matrix"]
        action = st.sidebar.selectbox("Choose One", actions)
        #st.markdown("____")
        #if st.checkbox('Train and Test Accuracies'):
        if action == 'Train and Test Accuracies':
            st.subheader('**Train Accuracy vs. Test Accuracy**')
            st.image('model_accuracy.JPG', width=700)
        #st.markdown("____")
        #if st.checkbox('\n\nEvaluation of the Model'):
        if action == 'Evaluation of the Model':
            st.subheader('**Evaluation of the Model**')
            st.image('model_metrics.JPG', width=700, height=700)
        #st.markdown("____")
        #if st.checkbox('\n\nConfusion Matrix'):
        if action == 'Confusion Matrix':
            st.subheader('**Confusion Matrix using all the classes**')
            st.image('model_cm.JPG', width=700)

    if choice == 'Recognize My Facial Expressions':
        st.title("Web Application For Facial Expression Recognition!")
        #html_choice = """ 
        #<marquee behavior="scroll" direction="left" width="100%;">
        #<h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        #</marquee><br>
        #"""
        #st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :hushed: :worried: :rage: :fearful:")
	st.markdown("____")
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        #if image if uploaded, display the progress bar and the image
        if image_file is not None:
            our_image = Image.open(image_file)
            st.markdown("**Original Image**")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(our_image)
        if image_file is None:
            st.error("No image uploaded yet")
            return

        # Face Detection
        task = ["Faces"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):
            if feature_choice == 'Faces':
                st.markdown("**Processing...\n**")

                #Progress bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i+1)
                #End of Progress bar

                result_img,result_faces,prediction = detect_faces(our_image)
                if st.image(result_img):
                    st.success("Found {} faces".format(len(result_faces)))

                    if prediction == 'Happy':
                        st.subheader("YeeY! You look **_Happy_** :smile: today, always be! ")
                    elif prediction == 'Angry':
                        st.subheader("You seem to be **_Angry_** :rage: today, just take it easy! ")
                    elif prediction == 'Disgust':
                        st.subheader("You seem to be **_Disgusted_** :rage: today! ")
                    elif prediction == 'Fear':
                        st.subheader("You seem to be **_Fearful_** :fearful: today, be couragous! ")
                    elif prediction == 'Neutral':
                        st.subheader("You seem to be **_Neutral_** today, wish you a happy day! ")
                    elif prediction == 'Sad':
                        st.subheader("You seem to be **_Sad_** :worried: today, smile and be happy! ")
                    elif prediction == 'Surprise':
                        st.subheader("You seem to be **_Surprised_** today! ")
                    else :
                        st.error("Your image does not seem to match the training dataset's images! Please try another image!")
                    
    elif choice == 'About':
        st.title("Web Application For Facial Expression Recognition!")
        #html_choice = """ 
        #<marquee behavior="scroll" direction="left" width="100%;">
        #<h2 style= "color: #000000; font-family: 'Raleway',sans-serif; font-size: 44px; font-weight: 700; line-height: 102px; margin: 0 0 24px; text-align: center; text-transform: uppercase;">Facial Expression Recognition Web Application </h2>
        #</marquee><br>
        #"""
        #st.markdown(html_choice, unsafe_allow_html=True)

        st.subheader(":smile: :worried: :fearful: :rage: :hushed:")
        st.markdown("____")
        st.markdown("**Dataset used for training:** https://www.kaggle.com/deadskull7/fer2013")
        st.subheader("About me:")
        st.markdown("**LinkedIn:** https://www.linkedin.com/in/subodh-lonkar-47662819b/")
        st.markdown("**GitHub:** https://github.com/learner-subodh")
        st.markdown("**Medium:** https://medium.com/@learner.subodh")

main()
