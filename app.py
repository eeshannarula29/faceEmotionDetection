import cv2
import consts
import numpy as np
import DataHelper as dh
from keras.models import load_model

model = load_model('emotions52.h5')

cap = cv2.VideoCapture(0)

while(True):

    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.rectangle(frame,(300,100),(800,600),(255,0,0),3)
    image = frame[100:500,100:500]
    image = cv2.resize(image,consts.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input = np.reshape(gray,consts.shape_for_nsamples(1))/255.0
    prediction = dh.getPrediction(model.predict(input)[0])



    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,prediction,(600,100),font, 4,(0,0,255),2,cv2.LINE_AA)
    # cv2.putText(frame,''.join(sentence),(100,600),font, 4,(0,0,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
