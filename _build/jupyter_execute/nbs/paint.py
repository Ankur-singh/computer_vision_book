#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Painting with OpenCV.
--------------------
click and drag your mouse to paint
Keys:
  r - reset the painting canvas
  q - exit
'''

import cv2
import numpy as np

drawing = False # true if mouse is pressed
img = None

# mouse callback function
def draw(event,x,y,flags,param):
    global drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
def main():
    global img
    img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image',img)
        
        k = cv2.waitKey(1)
        if k == ord('r'):
            img = np.zeros((512, 512, 3), np.uint8)
        if k == ord('q'):
            break

if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()

