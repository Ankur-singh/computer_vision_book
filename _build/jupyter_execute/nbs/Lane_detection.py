#!/usr/bin/env python
# coding: utf-8

# ### Lane Detection
# 
# Lane detection is a critical component of self-driving cars and autonomous vehicles. Once lane positions are obtained, the vehicle will know where to go and avoid the risk of running into other lanes or getting off the road. This can prevent the driver/car system from drifting off the driving lane.
# 
# The task that we are trying to perform is that of real-time lane detection in a video. [Click here](https://www.linkedin.com/posts/pranav-uikey-b49360140_machinelearning-selfdrivingcars-activity-6701729324074115073-Hd8d) to see how the final out will look like.
# 
# Lets start by importing all the necessary libraries

# In[4]:


import cv2
import numpy as np


# Next, we need to download the video file. We have already written a helper function to download any file from google drive. Lets import it from `utils.py`

# In[1]:


from utils import download_from_google_drive


# In[2]:


download_from_google_drive(file_id='1uqtYcogGc121DZK3eEfXz9yrXzIVtFAk', 
                           dest_path='./lane_detection.mp4', 
                           showsize=True)


# Now, you should have *lane_detection.mp4* in your present working directory.

# We will now define some helper functions. Everything we are doing here is already discussed in the previous notebooks.

# In[7]:


def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(0,650),(1250,650),(700,300)]
    ])
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask,polygons,1)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


# Let's Try things out on an image first then we will work on a video.

# In[5]:


cap = cv2.VideoCapture('lane_detection.mp4')
ret, image = cap.read()
cap.release()


# Above, we have taken the first frame of our video. Lets work with this frame first and then extent the application to work with video.

# In[8]:


canny_image = canny(image)
masked_image = roi(canny_image)

lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
line_image = display_lines(image, lines)
final_image = cv2.addWeighted(image, 0.7, line_image, 1, 1)

cv2.imshow('Original Image', image)
cv2.imshow('Canny Image', canny_image)
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# We apply canny edge detection, then select only the lower half of the image which has road in it (the sky and tree are not important). And then we passed our masked image to `cv2.HoughLinesP`, read more about it online. Finally, we used `cv2.addWeighted` to combine our original image and *line_image*. Its a pretty simple function and you can read more about it [here](https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#addweighted).
# 
# Great, we can now detect lane in a single image. Lets make it work with videos. 

# In[10]:


cap = cv2.VideoCapture('lane_detection.mp4')
while True:
    ret, image = cap.read()

    if not ret:
        break
    canny_image = canny(image)
    cropped_image = roi(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, minLineLength=40, maxLineGap=5)
    line_image = display_lines(image, lines)
    combined_image = cv2.addWeighted(image, 0.7, line_image, 1, 1)
    combined_image = cv2.resize(combined_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Lane Detection', combined_image)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()


# Making a real-time application is not very difficult. Also, the video is running at pretty high frame rate. The speed may vary depending upon your processor speed but the important point here is over computers are powerful enough to do all of it in couple of milli-seconds.
