#!/usr/bin/env python
# coding: utf-8

# # Face Detection
# 
# When you open camera app on your mobile phone to take someone's photo, it automatically detects all the faces in the image and makes a yellow box around all the detected faces. Not just camera app, face detection is everywhere. Facebook automatically detects all the faces in the images and suggests your names while tagging. 
# 
# As an exercise, try finding some more examples of face detection. You will be surprised.

# ## Haar Cascade Classifiers
# 
# In order to build face recognition application, we will use the built-in Haar cascade classifiers in OpenCV. These classifiers have already been pre-trained to recognize faces!
# 
# Building our own classifier is certainly outside the scope of this case study. But if we wanted to, we would need a lot of “positive” and “negative” images. Positive images would contain images with faces, whereas negative images would contain images without faces. Based on this dataset, we could then extract features to characterize the face (or lack of face) in an image and build our own classifier. It would be a lot of work, and very time consuming. Things are even more difficult for someone, who is a computer vision novice. Luckily, OpenCV will do all the heavy lifting for us.
# 
# **Working**
# 
# Haar-Cascade classifiers work by scanning an image from left to right, and top to bottom, at varying scale sizes. Scanning an image from left to right and top to bottom is called the “sliding window” approach. As the window moves from left to right and top to bottom, one pixel at a time, the classifier is asked whether or not it “thinks” there is a face in the current window, based on the parameters that are supplied to the classifier.
# 
# Lets write some code . . .

# In[1]:


import cv2
from utils import *


# In[42]:


img = cv2.imread('../images/friends.jpeg')
imshow('Image', img)


# I am a big fan of FRIENDS and hence big image, let's resize it so that its much eaiser to visualize. 

# In[43]:


h, w, _ = img.shape
img = cv2.resize(img, (int(w*0.8), int(h*0.8)))
imshow('Image', img)


# much better now. We took the initial dimensions and then multiplied them with 0.8. The `cv2.resize` function take an image and the new dimensions i.e 80% of the previous width & height.
# 
# **Note:** We switched the order of *width* and *height*. Because when we say `img.shape` it returns the dimensions as per numpy cordinate system, but when we are using `cv2.resize` function, its expects the dimensions in opencv cordinate system.
# 
# Lets now save the resized image for future use. 

# In[44]:


cv2.imwrite('images/friends2.jpeg', img)


# To detect faces in an image, we will have to convert the image into grayscale first. Infact, converting a color image to gray-scale is one of the most frequently used operation in computer vision. OpenCV has a handly little function for it.

# In[45]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow('Gray', gray)


# The first argument is always the image and second argument tells opencv about the current color space and the new color space to which the image is to be transformed.
# 
# You can use the same function to transform the image to other color spaces as well. By default, OpenCV uses BGR color space. To convert an image to RGB color space . .

# In[46]:


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow('RGB', img_rgb)


# to HSV color space . . . 

# In[47]:


img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imshow('HSV', img_hsv)


# You can read more about color spaces and their importances, [here](https://www.dynamsoft.com/blog/insights/image-processing-101-color-models/). 

# continuing with the problem in hand . . . 
# We have a black and white image now. Lets now talk about classifier.
# 
# Haar-Cascade classifiers are serialized as an XML file. You can easily load them using `cv2.CascadeClassifier()` method. This method take the path to the *xml* file as input and returns a classifier object.
# 
# **Note:** You can find *xml* files for a lot of other classifier in the official git repo of opencv, [here](https://github.com/opencv/opencv/tree/master/data/haarcascades) 
# 
# We have already download the *xml* file for detecting faces from the git repo. Its present inside data directory, lets try loading it

# In[17]:


face_cascade = cv2.CascadeClassifier('data/face.xml')


# Great, it ran without any error.
# 
# To detect actual faces in the image we make a call to the `detectMultiScale` method of our classifier. The method takes care of the entire face detection process. The method takes one required parameter, the image that he wants to find the faces in, followed by three optional arguments:
# 
# - `scaleFactor`:  How much the image size is reduced. A value of 1.05 indicates that the image will by reduced by 5%
# -  `minNeighbors`:  How many neighbors each window should have for the area in the window to be considered a face. The cascade classifier will detect multiple windows around a face. This parameter controls how many rectangles (neighbors) need to be detected for the window to be labeled a face.
# - `minSize`:  A tuple of width and height (in pixels) indicating the minimum size of the window. Bounding boxes smaller than this size are ignored.

# In[48]:


faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors=9, minSize=(28,28))


# The `detectMultiScale` method then returns *a list of tuples containing the bounding boxes of the faces* in the image. These bounding boxes are simply the *(x, y)* location of the face, along with the *width* and *height* of the box.

# In[49]:


faces


# our classifier has detected 6 faces in the image. We can draw one of them using `cv2.rectangle` function that we learned in the last notebook.

# In[51]:


x, y, w, h = faces[0]
upper_left_corner = (x, y)
lower_right_corner = (x+w, y+h)
color = (0, 255, 0)


# Lets draw our rectangle on the original image.

# In[52]:


img = cv2.rectangle(img, upper_left_corner, lower_right_corner, color, 2)
imshow('Image', img)


# so we have detected one face, now lets try to draw rectangles for all the faces that are detected

# In[50]:


color = (0, 255, 0)

for face in faces:
    x, y, w, h = face
    upper_left_corner = (x, y)
    lower_right_corner = (x+w, y+h)
    img = cv2.rectangle(img, upper_left_corner, lower_right_corner, color, 2)
    
imshow('Image', img)


# Amazing, we detected all the faces in the image. It looks a lot of code in the first go but actually its much simplier. I have written the complete code again, in a single cell. See if you can understand it now.

# In[37]:


img = cv2.imread('images/friends2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('data/face.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors=9, minSize=(28,28))

color = (0, 255, 0)

for face in faces:
    x, y, w, h = face
    img = cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    
imshow('Image', img)


# Its just ~10 lines of code. And the core logic is just ~3 lines. Sometimes, its really mind boggling how much you can achieve with just few lines of code.
# 
# It would be great to take this a step further and make it real time. We will take images from our webcam and detect faces, all in real-time.

# ## Real-time face detection
# 
# First we need to access our webcam. With opencv, its just a function call. 

# In[53]:


camera = cv2.VideoCapture(0)


# argument **0** means read from built-in or USB webcam. You can read a video file from the disk by simply passing its path, instead of **0**. 
# 
# Assuming that grabbing a reference to the video was successful, we can easily read the current frame by calling `read()` method of our *camera* object.

# In[54]:


ret, frame = camera.read()


# `camera.read()` returns two values. The first value is a boolean, indicating whether the frame capture was successful or not. The second value is the actual frame/image captured. Lets display the image we have captured

# In[55]:


imshow('Frame', frame)


# Once we know how to capture a single frame from the camera, we can easily loop over all frames in the video. At the most basic level, a video is simply a sequence of images put together, implying that we can read these frames one at a time.

# In[56]:


while True:
    ret, img = camera.read()
    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'): # press 'q' to quit
        break


# Whenever you write an infinite while loop, you have to implement the *break* condition. Here, the *break* condition looks a bit different. Lets break it down.
# 
# `cv2.waitKey` waits for a key press. Its takes one optional argument, *delay* in milliseconds. 0 is the special value that means “forever”.
# The function waitKey waits for a key event infinitely or for delay milliseconds, when it is positive. Here, we are using `cv2.waitKey(30)` it will display the frame for 30 ms, after which display will be automatically closed.
# 
# If in that 30 ms, you press any key then `cv2.waitKey` will return the interger corresponding to the key that was pressed. You can simply *&* the returned value with *0xff*. *0xff* is a hexadecimal constant which is 11111111 in binary. By using bitwise AND (&) with this constant, it leaves only the last 8 bits of the original (in this case, whatever cv2.waitKey(0) is). 
# 
# Once you have the key value, you can check if it was *q* or not. If the key pressed was *q* then you can break the loop. The `ord()` function in Python accepts a character as an argument and returns the unicode code point representation of the passed argument. For example, in the above code, ord('q') returns 113 which is a unicode code point value of character 'q'.
# 
# It might look like a lot to remember and understand but, trust me, its very easy and you will get use to it pretty soon. 
# 
# Final thing! In programming, whenever you access an I/O device like camera, USB, database and even files; you will have to close it explicitely when you are done using it. Here, we are capturing frames from our webcam. So, we will have to close it once we are done.

# In[61]:


camera.release()
cv2.destroyAllWindows()


# Its a general practive to release the camera and then destroy all the windows. So, you will always see these commands used together. 
# 
# Lets put everything back together. See if you understand it now

# In[ ]:


camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'): # press 'q' to quit
        break
        
camera.release()
cv2.destroyAllWindows()


# This is all the code you need to read images from your webcam and display them. You can also save the images by calling `cv2.imwrite`. 
# 
# We already know how to find faces in a single image. Since, inside the loop we have access to individual frames we easily use the same code from above the detect faces in each frame and then display them. Lets see it in action 

# In[63]:


color = (0, 255, 0)
face_cascade = cv2.CascadeClassifier('data/face.xml')
camera = cv2.VideoCapture(0)

while True:
    ret, img = camera.read()
    if ret:
        ## Code to detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15, minNeighbors=9, minSize=(28,28))

        for face in faces:
            x, y, w, h = face
            img = cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'): # press 'q' to quit
        break
        
camera.release()
cv2.destroyAllWindows()


# We won't be explaining the code again, because there is nothing new. Spend a minute or two read the complete code. If you have any doubt you can ask your mentor.
# 
# **Note:** One thing to notice here is, how easy it is to make a real-time application if you have the code to process an image. This is the case with all the computer vision applications that we build, we start with a single image and write all the code for it. Finally, to make it a real-time application, we simple grab the camera object and wrap an infinte while loop around it.
# 
# This is all the extra code you need to make something real-time in opencv
# 
# ```python
# camera = cv2.VideoCapture(0)
# 
# while True:
#     ret, img = camera.read()
#     if ret:
#         process_img(img) # only this function is updated, depending upon the application
#         cv2.imshow('video',img)
# 
#     k = cv2.waitKey(30) & 0xff
#     if k == ord('q'): # press 'q' to quit
#         break
#         
# camera.release()
# cv2.destroyAllWindows()
# 
# ```
# 
# Its amazing, right?! Don't worry if you did not understand everything now, we will be building many applications throughout the course. You will easily get accustom to it.

# ## Questionaire
# - How to resize images?
# - Name a few color spaces and also how to convert images from one color space to another?
# - What are haar cascade classifier and how they work?
# - How to read video input: webcam or mp4 from disk?
# - Difference between `cv2.waitkey(0)`, `cv2.waitkey(30)` and `cv2.waitkey(100)`
# - Why is closing I/O devices important?
# - Which command is used to release camera?
# 
# If you don't know the answer to any of these question, then please read the notebook again or ask your mentor. There are some questions that your are suppose to google and read to know more.
# 
# In the next notebook, we will learn about image transformations like rotation, scaling, translation, etc.
