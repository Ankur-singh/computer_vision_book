#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Authenticate Employees
----------------------
Office In time scanner. 
Green color shows authorized person and red shows unauthorized person.
Both scanning timing and person's name gets saved in a new file.

Keys:
  q - exit
'''

import os
import cv2
import numpy as np
from datetime import datetime, date

# Main library for working with barcodes
from pyzbar.pyzbar import decode


def main(record=True):
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 

    # For saving the recording
    if record :
        result = cv2.VideoWriter('QRCode.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 

    # To create new file for daywise IN-TIME data
    today = date.today() 
    writepath = str(today) + '.txt' 
    mode = 'a' if os.path.exists(writepath) else 'w'

    # Authorised users list
    myDataList = ['Pranav Uikey']


    while True:
        success, img = cap.read()

        # Captures current time 
        now = datetime.now().time()
        time = str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)

        # Decode the QR Code(Main loop)
        for barcode in decode(img):
            myData = barcode.data.decode('utf-8')
            if myData in myDataList:
                myOutput = myData + '@' + time
                myColor = (0, 255, 0)
            else:
                myOutput = myData + '@' + time
                myColor = (0, 0, 255)

            # Write the IN TIME in daywise file
            with open(writepath, mode) as f:
                f.write(myData + ',' + time + '\n')

            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, myColor, 5)
            pts2 = barcode.rect
            cv2.putText(img, myOutput, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)

        cv2.imshow('Result', img)
        if record:
            result.write(img)

        k = cv2.waitKey(30) & 0xff
        if k == ord('q'): 
            f.close()
            break

    # Release all resources
    if record:
        result.release()
    cap.release()

if __name__ == "__main__":
    print(__doc__)
    main()
    cv2.destroyAllWindows()

