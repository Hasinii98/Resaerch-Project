import numpy as np
#from  flask import Flask
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential#to build sequencial neural network
from tensorflow.keras.layers import LSTM, Dense #to perform action detection
from tensorflow.keras.callbacks import TensorBoard#to monitor and traise model

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rake_nltk import Rake 
import csv


import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from tensorflow import keras
import pickle 
import webbrowser
import cv2

from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

     
@app.route('/months',methods=['POST'])
def months():


    poses=np.array(['January','February','March','April','May','June','July','August','September','October','November','December'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('months.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('months_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('months_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)

            if x == 'January':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/January.mp4?alt=media&token=ad1da350-0dc5-49db-8b10-ea52c7efa2d0'
            elif x == 'February':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/February.mp4?alt=media&token=a1e04883-5823-4bc4-9707-6f66d0942a3f'
            elif x == 'March':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/March.mp4?alt=media&token=3611ffe1-92bc-4f35-b966-7dfaf4415eab'
            elif x == 'April':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/April.mp4?alt=media&token=6700da74-c716-4f39-87ce-7471c052bf9b'
            elif x == 'May':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/May.mp4?alt=media&token=d4de0c70-a11f-42f9-b247-c426f4231e6b'
            elif x == 'June':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/June.mp4?alt=media&token=2b4c9615-9876-49b4-b0f7-7546065e833e'
            elif x == 'July':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/July.mp4?alt=media&token=d6f84173-fe5b-4d4f-9b30-0a562510cdcc'
            elif x == 'August':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/August.mp4?alt=media&token=a5d31164-d367-4e5e-b09e-343ee11ed225'
            elif x == 'September':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/September.mp4?alt=media&token=f9b4b646-c730-46ef-8702-c48d37c36459'
            elif x == 'October':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/October.mp4?alt=media&token=c1c1fb33-abba-4447-bc87-c63f5d7be0ae'
            elif x == 'November':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/November.mp4?alt=media&token=270d2679-267a-4a59-a768-a49aafddff55'  
            elif x == 'December':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/December.mp4?alt=media&token=c8fc67d6-6e6e-4692-8ab6-650c1df67b70'  
        

        return (render_template('index.html',value=x, mode=mode))

@app.route('/numbers',methods=['POST'])
def numberpredict():

    json_file = open("model-bw_numbers.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw_numbers.h5")
    print("Loaded model from disk")


    cap = cv2.VideoCapture(0)

    #categories = {A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'FIVE', G: 'G', H: 'H', I: 'I', J: 'J', K: 'K', L: 'L',
    #            M: 'M', N: 'N', O: 'O', P: 'P', Q: 'Q', R: 'R', S: 'S', T: 'T', U: 'U', V: 'V', W: 'W', X: 'X',
    #           Y: 'Y', Z: 'Z'}

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {
                    '0': result[0][0], 
                    '1': result[0][1], 
                    '2': result[0][2], 
                    '3': result[0][3], 
                    '4': result[0][4], 
                    '5': result[0][5], 
                    '6': result[0][6],
                    '7': result[0][7],
                    '8': result[0][8],
                    '9': result[0][9],
                    '10': result[0][10]
                    
                                }
        
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 5)    
        cv2.imshow("Frame", frame)
        result = prediction[0][0]
        
        with open('result_number','wb') as f:
                pickle.dump(result,f)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
            
    
    cap.release()
    cv2.destroyAllWindows()  
            

    with open('result_number','rb') as f:
        x = pickle.load(f)
        #print(x)

        if x == '0':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/0.mp4?alt=media&token=bb11377a-b89d-460f-8b99-e592a97a4f9e'
        elif x == '1':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/1.mp4?alt=media&token=a51a4987-96f3-4fb8-8f66-c67c484bda8a'
        elif x == '2':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/2.mp4?alt=media&token=99883d08-126f-40a3-965e-982ea46f08e2'
        elif x == '3':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/3.mp4?alt=media&token=7300eba9-81c7-45de-8af4-9ef87bb3b659'
        elif x == '4':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/4.mp4?alt=media&token=05b089ad-6929-4aa6-90b4-8838d5578e62'
        elif x == '5':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/5.mp4?alt=media&token=10a49e72-8605-4e94-a2f6-1dcc519bfa88'
        elif x == '6':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/6.mp4?alt=media&token=75baa9c2-34ea-4c09-ac78-3a228e4821fa'
        elif x == '7':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/7.mp4?alt=media&token=8967d9b3-c039-4851-a3ed-1d2e8ebb5d7a'
        elif x == '8':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/8.mp4?alt=media&token=c4191a28-4636-47df-b06d-dfef8f20e095'
        elif x == '9':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/9.mp4?alt=media&token=35aad1fc-63a2-4ad3-bc0b-2825f1b50057'
        elif x == '10':
                mode = 'https://www.google.com/'
        
    return (render_template('index.html',value=x, mode=mode))


@app.route('/days',methods=['POST'])
def days():


    poses=np.array(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('days-dw.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('days_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('days_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)
            
            if x == 'Monday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Monday.mp4?alt=media&token=66e46e94-bc6a-4c07-a59f-51daae739c25'
            elif x == 'Tuesday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Tuesday.mp4?alt=media&token=a704f042-1152-4681-bb3b-6319b175702a'
            elif x == 'Wednesday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Wednesday.mp4?alt=media&token=e3ffba2d-1483-47ea-b4ba-4161eea99854'
            elif x == 'Thursday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Thursday.mp4?alt=media&token=54ad0a3b-7921-4925-876a-932b6b7b5bc6'
            elif x == 'Friday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Thursday.mp4?alt=media&token=54ad0a3b-7921-4925-876a-932b6b7b5bc6'
            elif x == 'Saturday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Saturday.mp4?alt=media&token=dc2c10dc-0eaf-44de-b1aa-19c5806c199d'
            elif x == 'Sunday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Sunday.mp4?alt=media&token=fa62ea62-9288-45e8-8918-284bd3afad05'
                    

        return (render_template('index.html',value=x, mode=mode))



@app.route('/prediction',methods=['POST'])
def prediction():

    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")


    # In[40]:


    cap = cv2.VideoCapture(0)


    # Category dictionary

    # In[41]:


    #categories = {A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'FIVE', G: 'G', H: 'H', I: 'I', J: 'J', K: 'K', L: 'L',
    #            M: 'M', N: 'N', O: 'O', P: 'P', Q: 'Q', R: 'R', S: 'S', T: 'T', U: 'U', V: 'V', W: 'W', X: 'X',
    #           Y: 'Y', Z: 'Z'}


    # In[ ]:


    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {'A': result[0][0], 
                    'B': result[0][1], 
                    'C': result[0][2],
                    'D': result[0][3],
                    'E': result[0][4],
                    'F': result[0][5],
                    'G': result[0][6],
                    'H': result[0][7],
                    'I': result[0][8],
                    'J': result[0][9],
                    'K': result[0][10],
                    'L': result[0][11],
                    'M': result[0][12],
                    'N': result[0][13],
                    'O': result[0][14],
                    'P': result[0][15],
                    'Q': result[0][16],
                    'R': result[0][17],
                    'S': result[0][18],
                    'T': result[0][19],
                    'U': result[0][20],
                    'V': result[0][21],
                    'W': result[0][22],
                    'X': result[0][23],
                    'Y': result[0][24],
                    'Z': result[0][25]          
                    
                                }
        
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 5)    
        cv2.imshow("Frame", frame)
        result = prediction[0][0]
        #print(result)


        with open('alphabet_results','wb') as f:
            pickle.dump(result,f)
        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break

    cap.release()
    cv2.destroyAllWindows()  
        

    with open('alphabet_results','rb') as f:
        x = pickle.load(f)
        #print(x)
        
        if x == 'A':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/A.mp4?alt=media&token=217c5107-821f-420d-8e22-790426c0a011'   
        elif x == 'B':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/B.mp4?alt=media&token=8757ab79-2427-406e-8065-e7f4e568cc27'   
        elif x == 'C':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/C.mp4?alt=media&token=ce0cf661-b7d5-4765-bdb6-8b1a0c12d341'
        elif x == 'D':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/D.mp4?alt=media&token=0e668a08-03e8-446c-8d07-d76575096cc9'
        elif x == 'E':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/E.mp4?alt=media&token=e3c61d70-31fc-4193-b86f-f409f6fff8c7'
        elif x == 'F':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/F.mp4?alt=media&token=b2da604d-1097-4d8b-a9b9-83eaa9ef69b6'
        elif x == 'G':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/G.mp4?alt=media&token=0c0b702f-ada7-43d8-931e-4b8d72791efa'
        elif x == 'H':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/H.mp4?alt=media&token=19b18842-1643-47ee-806b-4717fda069df'
        elif x == 'I':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/I.mp4?alt=media&token=9a648fda-1a96-4ff9-bcae-52ecf9e3e5df'
        elif x == 'J':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/J.mp4?alt=media&token=698e50f1-fd6a-4c1c-b0c6-c98e8011d6de'
        elif x == 'K':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/K.mp4?alt=media&token=4247abab-3f4c-4aa5-b533-040613308099'
        elif x == 'L':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/L.mp4?alt=media&token=ab9b2daf-be9f-4848-b92e-a46dac2ddfe'
        elif x == 'M':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/M.mp4?alt=media&token=18286a59-e791-48f5-9923-93004f0a522b'
        elif x == 'N':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/N.mp4?alt=media&token=bc519836-4dd3-44c7-8a57-267b1ae2420c'
        elif x == 'O':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/O.mp4?alt=media&token=44879657-9b16-4acc-96ba-6e4631831109'
        elif x == 'P':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/P.mp4?alt=media&token=5de29802-aa4d-4a9f-bf1d-d2a40cbf654c'
        elif x == 'Q':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Q.mp4?alt=media&token=d60b2c7a-5d9e-4a3c-a2d1-fc5d024ce0be'
        elif x == 'R':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/R.mp4?alt=media&token=7b5c8d06-eb7b-457f-880e-3afc9e057026'
        elif x == 'S':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/S.mp4?alt=media&token=a15ff787-be7a-4e11-878c-7f02d24b9ea6'
        elif x == 'T':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/T.mp4?alt=media&token=aa671d0a-9e40-48d4-b8f4-6371976869fa'
        elif x == 'U':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/U.mp4?alt=media&token=1d421047-419a-4ad8-8b60-92dc01a93a08'
        elif x == 'V':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/V.mp4?alt=media&token=8e9f2070-ec4b-46eb-bb5a-7d9cf7f2f850'
        elif x == 'W':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/W.mp4?alt=media&token=4cb3e1df-a22c-4acc-9599-31fb4951e027'
        elif x == 'X':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/X.mp4?alt=media&token=b0558f0a-ee6b-4c1d-b7ff-dc600525fd8f'
        elif x == 'Y':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Y.mp4?alt=media&token=108f86ea-aa02-4c08-a10c-de97233cca38'
        elif x == 'Z':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Z.mp4?alt=media&token=3c76e471-0013-4213-90c0-d303b77bff66'
        
        
    
    return (render_template('index.html',value=x, mode =mode))

        
    
    return (render_template('index.html',value=x))

    

#res[np.argmax(res)] > threshold


@app.route('/colours',methods=['POST'])
def colours():


    poses=np.array(['Black','Blue','Green','Red','White','Yellow'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('colour-dw.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('colour_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('colour_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)

            if x == 'Black':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Black.mp4?alt=media&token=82a869ef-36cf-4bb6-8f02-0e31fbf0f9b3'   
            elif x == 'Blue':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Blue.mp4?alt=media&token=0a7dde13-0930-48d3-bc4b-bd30b4b3af51'   
            elif x == 'Green':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Green.mp4?alt=media&token=75db0f59-6f30-4855-836d-81d8241bbf13'
            elif x == 'Red':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Red.mp4?alt=media&token=51cc51fe-10d9-474c-b7f5-7b0f5027cfbf'
            elif x == 'White':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/White.mp4?alt=media&token=3b50cbf8-ced7-4908-87e7-560979c76592'
            elif x == 'Yellow':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Yellow.mp4?alt=media&token=6c7310ae-6e28-44a6-b27e-b7593fe4a1b5'
            
        

        return (render_template('index.html',value=x, mode=mode))

        #$env:FLASK_APP = "Sign language detection.py"
    #flask run