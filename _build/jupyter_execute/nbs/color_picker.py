#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
Color Picker in OpenCV.
--------------------
Keys:
  q - exit
'''

import cv2
import numpy as np

def nothing(x): pass

def main():
    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        # get current positions of trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        img[:] = [b, g, r]
    print('The RGB value of selected color is {}, {}, {}'.format(r,g,b))
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()

