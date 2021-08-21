import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
model=load_model('facial.h5')
cv.ocl.setUseOpenCL(False)
dic={0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
cap=cv.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    bounding_box=cv.CascadeClassifier(r'C:\Users\naman\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    grey_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=bounding_box.detectMultiScale(grey_frame,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y-50),(x+w, y+h+10),(255,0,0),2)
        roi=grey_frame[y:y+h,x:x+w]
        cropped=np.expand_dims(np.expand_dims(cv.resize(roi,(48,48)),-1),0)
        pred=model.predict(cropped)
        maxindex=int(np.argmax(pred))
        cv.putText(frame,dic[maxindex],(x+20,y-60),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv.LINE_AA)
    cv.imshow('Video',cv.resize(frame,(1200,860),interpolation=cv.INTER_CUBIC))
    if cv.waitKey(1)&0xFF==ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break