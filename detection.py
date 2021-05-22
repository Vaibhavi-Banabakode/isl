from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic
def detect(frame):
    cv2.circle(frame,(300,300),radius=3,color=(255,0,0),thickness=2)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
                
                # Make Detections
        results = holistic.process(image)
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                            )

                # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                )
        if ((results.right_hand_landmarks!=None) and (results.left_hand_landmarks!=None)):
            with open('isl_hand.pkl','rb') as f:
                model=pickle.load(f)
                    #both_hand_coord_length = len(results.right_hand_landmarks.landmark)+len(results.left_hand_landmarks.landmark)
            right = results.right_hand_landmarks.landmark
            right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
                    # Extract Face landmarks
            left = results.left_hand_landmarks.landmark
            left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
            
                    # Concate rows
            row = right_row+left_row
                    #print('both handrow:',row)
                    # Make Detections
            X = pd.DataFrame([row])
            sign_language_class = model.predict(X)[0]
            sign_language_prob = model.predict_proba(X)[0]
            print(sign_language_class, sign_language_prob)
                    
                    # Display Class
            cv2.putText(image, 'Detection'
                                , (100,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, sign_language_class
                                , (90,50), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
            cv2.putText(image, 'Accuracy'
                                , (10,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)],2))
                                , (10,50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        elif (results.right_hand_landmarks!=None):
            with open('right_hand.pkl','rb') as f:
                model=pickle.load(f)
                    #single_hand_coord_length=len(results.right_hand_landmarks.landmark)
            right = results.right_hand_landmarks.landmark
            right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
                    # Extract Face landmarks
                    
            row = right_row
                    #print('single hand row:',row)

                    # Make Detections
            X = pd.DataFrame([row])
            sign_language_class = model.predict(X)[0]
            sign_language_prob = model.predict_proba(X)[0]
            print(sign_language_class, sign_language_prob)
                    
            
                    # Display Class
            cv2.putText(image, 'Detection'
                                , (100,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, sign_language_class
                                , (90,50), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
            cv2.putText(image, 'Accuracy'
                                , (10,20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)],2))
                                , (10,50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return image
        
    #cap.release()
    cv2.destroyAllWindows()

        
