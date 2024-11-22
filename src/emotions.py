import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HEARTBEAT_API_URL = 'http://127.0.0.1:5000/vitals'
MAIL_API_URL = 'http://127.0.0.1:8000' 

def get_datetime():
    now = datetime.now()
    current_datetime_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    return current_datetime_formatted 

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def detect_emotions(state, mode='train'):
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    if mode == "train":
        model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.0001), 
              metrics=['accuracy'])
        model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size
        )
        plot_model_history(model_info)
        model.save_weights('model.h5')

    elif mode == "display":
        model.load_weights('model.h5')

        cv2.ocl.setUseOpenCL(False)

        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        output_filename_raw = f'emotion_videos/raw/emotion_raw.{get_datetime()}.avi'
        output_filename_detected = f'emotion_videos/detected/emotion_detected.{get_datetime()}.avi'  
        fps = 20.0 

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width,frame_height)

        raw_out = cv2.VideoWriter(output_filename_raw, fourcc, fps, frame_size)
        detected_out = cv2.VideoWriter(output_filename_detected, fourcc, fps, frame_size)

        last_print_time = time.time()  

        while True:
            ret, frame = cap.read()
            raw_out.write(frame)
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                emotion_detected = emotion_dict[maxindex]
                
                if time.time() - last_print_time >= 5:
                    try:
                        response = requests.get(HEARTBEAT_API_URL)
                        heart_beat = response.json().get('heartbeat', 'Error fetching heartbeat')
                        blood_pressure = response.json().get('blood_pressure', 'Error fetching heartbeat')
                        if emotion_detected=="Happy":
                            blood_pressure="Low"
                        elif emotion_detected=="Neutral":
                            blood_pressure="Low"
                        elif emotion_detected=="Anger":
                            blood_pressure="High"
                        elif emotion_detected=="Sad":
                            blood_pressure="Medium"
                        elif emotion_detected=="Fearful":
                            blood_pressure="High"
                        print("Emotion Detected: ", emotion_detected, "\nHeartbeat: ", heart_beat, "\nState: ", state,"\nBlood Pressure: ", blood_pressure)

                        vitals_data = {
                            'state': state,
                            'emotion': emotion_detected,
                            'heartbeat': heart_beat,
                            'message' : "", 
                            'blood_pressure' : blood_pressure,
                        }

                        if((emotion_detected == "Sad" or emotion_detected == "Fearful") and state == "Isolated"):
                            vitals_data['message'] = "abnormal behaviour detected (sad or fearful)"
                            mail_response = requests.post(MAIL_API_URL, data=vitals_data)
                            if mail_response.status_code == 200:
                                print("Vitals sent to email!")
                            else:
                                print("Failed to send vitals to email.")

                         


                    except requests.exceptions.RequestException as e:
                        print("Error fetching heartbeat:", e)
            
                    last_print_time = time.time()

                cv2.putText(frame, emotion_detected, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            detected_out.write(frame)   
            cv2.imshow('Video', cv2.resize(frame, (500, 500), interpolation=cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()