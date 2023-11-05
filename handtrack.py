import cv2
import mediapipe as mp
import time

video=cv2.VideoCapture(1)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpdraw=mp.solutions.drawing_utils

ctime=0
ptime=0

while True:
    ret,img=video.read()
    imgrgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(imgrgb)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for hndmarks in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, hndmarks ,mpHands.HAND_CONNECTIONS)

    ctime=time.time()
    fps=1/(ctime - ptime)
    ptime=ctime

    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
    cv2.imshow("frame",img)     
    k=cv2.waitKey(1)
    if k==ord("q"):
        break


video.release()
cv2.destroyAllWindows()