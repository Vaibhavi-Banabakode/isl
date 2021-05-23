import csv
import os
import numpy as np
import isl_media
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic
x=isl_media.capture()
if x==42:
    landmarks = ['class']
    for val in range(1, x+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    print(landmarks)
elif x==21:
    landmarks = ['class']
    for val in range(1, x+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    print(landmarks)
def csvfile():
    csv_file_name=input('Enter csv file name:')
    with open(csv_file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
def single_hand():
    class_name=input('Enter class name:')
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            #                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            #                       mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            #                      )

            # Export coordinates
            try:
                # Extract Pose landmarks
                right = results.right_hand_landmarks.landmark
                right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
                
                # Extract Face landmarks
                #left = results.left_hand_landmarks.landmark
                #left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
                
                # Concate rows
                row = right_row
                
                # Append class name 
                row.insert(0, class_name)
                
                # Export to CSV
                with open('one.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 
                
            except:
                pass
                            
            cv2.imshow('Single Hand Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
def both_hands():
    class_name=input('Enter class name:')
    cap = cv2.VideoCapture(0)
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                   mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                  )

            # Export coordinates
            try:
                # Extract Right hand landmarks
                right = results.right_hand_landmarks.landmark
                right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
                
                # Extract left hand landmarks
                left = results.left_hand_landmarks.landmark
                left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
                
                # Concate rows
                row = right_row+left_row
                
                # Append class name 
                row.insert(0, class_name)
                
                # Export to CSV
                with open('two.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 
                
            except:
                pass
                            
            cv2.imshow('Both Hand Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    csvfile()
    if x==42:
        both_hands()
    elif x==21:
        single_hand()