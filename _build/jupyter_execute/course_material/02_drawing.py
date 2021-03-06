#!/usr/bin/env python
# coding: utf-8

# # More on pixels
# 
# In the first part of the notebook, we will learn about accessing and manipulating pixel values. In the second part, we will learn to draw different shapes on an image, using opencv.

# In[1]:


import cv2
import numpy as np


# ## Accessing pixel values

# In[2]:


img = cv2.imread('images/yoda.jpeg')


# Whenever we read an image using `cv2.imread()`, it returns a numpy array. Access values from numpy array is super easy. You can access a single pixel by simply passing its cordinates.

# In[3]:


pixel_00 = img[0, 0]
pixel_00


# To access a region of an image, we can use slicing. For example, this is how we can access the upper-left region of the image

# In[4]:


upper_left_5 = img[:5, :5]
upper_left_5


# to select the lower-right region you can do `img[-5:0, -5:0]`. If you are new to numpy, we would recommend you to checkout [indexing documentation](https://numpy.org/doc/stable/reference/arrays.indexing.html).
# 
# 5 x 5 is a very small region to display, so lets select a bigger region.

# In[5]:


upper_left_150 = img[:150, :150]
upper_left_150.shape


# You can think of this as a 150 x 150 image with 3 channels. Lets display it now.

# In[6]:


cv2.imshow('Upper Left corner', upper_left_150)
cv2.waitKey(0)
cv2.destroyAllWindows()


# This process of selecting a small part of an images is also know as **cropping**. It is one of the easiest transformation that you can apply to your images.

# ## Manipulating images
# 
# Again, since its just a numpy array, we can simply assign a new value to the pixels. Lets select the upper-left corner and make it blue.

# In[7]:


img[:150, :150] = (255, 0, 0)


# **Note:** OpenCV by default stores pixels in BGR format, not RGB format. So, (255, 0, 0) is blue not red.
# 
# when we assign a tuple (of 3 values, representing a color) to the selected region, it is automatically broadcasted to all the pixel in the selected region. This concept is called broadcasting and if you are listening it for the first time, then we recommend you to check [the docs](https://numpy.org/doc/stable/user/basics.broadcasting.html).
# 
# Lets have a look at our image after manipulation.

# In[8]:


cv2.imshow('Colored corner', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# We have called these 3 lines multiple times. Lets put them inside a function for future use. 

# In[10]:


def imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# **Exercise:**
# 
# - Try selecting all the 4 corner and color them with different colours.
# - Also, select a 100 x 100 region, in the center of the image and make it white.

# ## Drawing
# 
# Using NumPy array slices, we were able to draw a square on our image. But what if we wanted to draw a single line? Or a circle? NumPy does not provide that type of functionality – it’s only a numerical processing library after all!
# 
# Luckily, OpenCV provides convenient, easy-to-use methods to draw shapes on an image. We’ll review the three most basic methods to draw shapes: `cv2.line`, `cv2.rectangle`, and `cv2.circle`.
# 
# Lets start by defining our canvas. Since images is nothing but a numpy arrays, we can manually define a numpy array with all zeros, and use it as our canvas.

# In[11]:


canvas = np.zeros((500, 500, 3))


# to see how our canvas looks we will use the `show()` function we defined above.

# In[12]:


imshow('canvas',canvas)


# why is our canvas **black**? Think for a second. What are the pixel values? 
# 
# Lets draw some shapes . . .

# ### Lines
# 
# In order to draw a line, we make use of the `cv2.line` method. The first argument to this method is the image we are going to draw on. In this case, it’s our canvas. The second argument is the starting point of the line. We will start our line, at point (0, 10). We also need to supply an ending point for the line (the third argument). We will define our ending point to be (500, 10). The last argument is the color of our line, we will use blue color. 
# 
# Lets see `cv2.line` in action

# In[13]:


blue = (255, 0, 0)
canvas = cv2.line(canvas, (0, 10), (500, 10), blue)


# Here, we drew a blue line from point (0, 10) to point (500, 10). Lets draw another line with thickness of 2px

# In[14]:


green = (0, 255, 0)
canvas = cv2.line(canvas, (0, 30), (500, 30), green, 2)


# time to look at our image . . . 

# In[15]:


imshow('canvas', canvas)


# As you can see, drawing a line is pretty easy. Specify the canvas, start & end point, color, and thickness (optional). 
# 
# **Note:** Whenever using `cv2.line`, `cv2.rectangle` and `cv2.circle` make sure you use the **OpenCV cordinate system**, not numpy system. Because, [5, 10] in numpy system means 5th row, 10th column. Where as in opencv system, it means 5th column (x-axis) and 10th row (y-axis).

# ### Rectangle
# 
# To draw a rectangle, we make use of the `cv2.rectangle` method. `cv2.rectangle` is very similar to `cv2.line`. The only difference is, instead of passing start and end points of the line, we will pass upper-left and lower-right corner of the rectangle.

# In[16]:


red = (0, 0, 255)
canvas = cv2.rectangle(canvas, (0, 50), (300, 100), red, 2)


# In[17]:


imshow('canvas', canvas)


# we have only drawn the outline of a rectangle. To draw a rectangle that is “filled in”, like when using NumPy array slices, we can pass negative thickness.

# In[18]:


red = (0, 0, 255)
canvas = cv2.rectangle(canvas, (10, 70), (310, 120), red, -1)


# In[19]:


imshow('canvas', canvas)


# ### Circles
# 
# Drawing circles is just as simple as drawing rectangles, but the function arguments are a little different. To draw a circle, we need two things: center and radius.

# In[20]:


center = (300, 300)
radius = 20
canvas = cv2.circle(canvas, center, radius, blue, 2)


# In[21]:


imshow('canvas', canvas)


# To draw a fill-in circle, just change  the thickness from 2 to -1.
# 
# Lets make some concentric circles.

# In[22]:


canvas = np.zeros((500, 500, 3))
(centerX, centerY) = (canvas.shape[1] // 2, canvas.shape[0] // 2)
red = (0, 0, 255)
for r in range(0, 175, 25):
    canvas = cv2.circle(canvas, (centerX, centerY), r, red)


# In[23]:


imshow('canvas', canvas)


# This is great, we have learned how to draw lines, rectangles and circles in OpenCV. Lets do some fun projects based on this newly acquired skill.
# 
# **Exercise:** 
# - Make 640 x 640 canvas and make it look like a chess board with alternate white and black squares.
# - Create a canvas, randomly select a point (`center`), randomly select a color (`c`), and finally, randomly select a radius (`r`). Now, make a circle center at `center`, of radius `r` and fill-in color `c`. Repeat it 25 times. Now, does it look **psychedelic**? 

# ## Questionaire
# 
# - Numpy indexing
# - Broadcasting
# - OpenCV uses which format BGR or RGB? What difference does it make?
# - Which cordinate system is used by `cv2.line`, `cv2.rectangle` and `cv2.circle`; Numpy or OpenCV?
# - How to drawing a line, rectangle and circle?
# 
# Make sure you know answers to all these questions before you move ahead in the course.
# 
# Its time, we should now start building some useful applications. In the next notebook, we will build a face detector. 
