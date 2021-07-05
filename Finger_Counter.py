import mediapipe as mp
import os
import numpy as np
import cv2
import uuid
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.6) as hands:
    intializer = 0
    cap = cv2.VideoCapture(0)
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if cap.isOpened() == False:
        print("Error in opening video stream or file")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            #setting image writables false and perform detection and then set back to true for perforamnce improvement u might wanna research on that xD
            img.flags.writeable = False
            result = hands.process(img) #1st process aka detections
            img.flags.writeable = True
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            #render result
            if result.multi_hand_landmarks:
                normalizedLandmark_1 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pixelCoordinatesLandmark_1 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_1.x, normalizedLandmark_1.y, frameWidth, frameHeight))[1]
                normalizedLandmark_2 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                pixelCoordinatesLandmark_2 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_2.x, normalizedLandmark_2.y, frameWidth, frameHeight))[1]
                normalizedLandmark_3 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                pixelCoordinatesLandmark_3 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_3.x, normalizedLandmark_3.y, frameWidth, frameHeight))[1]
                normalizedLandmark_4 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                pixelCoordinatesLandmark_4 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_4.x, normalizedLandmark_4.y, frameWidth, frameHeight))[1]
                normalizedLandmark_5 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pixelCoordinatesLandmark_5 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_5.x, normalizedLandmark_5.y, frameWidth, frameHeight))[1]
                normalizedLandmark_6 = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                pixelCoordinatesLandmark_6 = list(mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark_6.x, normalizedLandmark_6.y, frameWidth, frameHeight))[1]
                    #print(list(pixelCoordinatesLandmark_1)[1],list(pixelCoordinatesLandmark_2)[1])
                    #print(normalizedLandmark_1,normalizedLandmark_2)
                if(pixelCoordinatesLandmark_1 < pixelCoordinatesLandmark_2):
                    intializer = 1
                    if(pixelCoordinatesLandmark_3 < pixelCoordinatesLandmark_4):
                        intializer = 2
                        if(pixelCoordinatesLandmark_5 < pixelCoordinatesLandmark_6):
                            intializer = 3
                        
                cv2.putText(img,str(intializer),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),2)
                        

                for num,hand in enumerate(result.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color = (0,255,255),thickness=2,circle_radius=4),mp_drawing.DrawingSpec(color = (0,65,255),thickness=2,circle_radius=2))
            else :
                cv2.putText(img,str(intializer),(100,100),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),2)
            cv2.imshow('Frame',img)
            # Press esc to exit
            if cv2.waitKey(20) == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
   
