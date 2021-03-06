#!/usr/bin/env python
# coding: utf-8

# In[32]:


import cv2
import numpy as np
from utils import imshow


# # Smoothing and blurring

# We all know what blurring is. It’s what happens when your camera takes a picture out of focus. Sharper regions in the image lose their detail, normally as a disc or circular shape.
# 
# Practically, this means that each pixel in the image is mixed in with its surrounding pixel intensities. This “mixture” of pixels in a neighborhood becomes our blurred pixel.
# 
# While this effect is usually unwanted in our photographs, it’s actually quite helpful when performing image processing tasks. In fact, many image processing and computer vision functions, such as thresholding and edge detection, perform better if the image is first smoothed or blurred.
# 
# **Note:** Sometimes the process is also called **filtering** because we use filters (i.e kernels) to smooth the image.
# 
# There are many techniques of smoothing your images. Lets learn them one by one using *blurring.png* image.

# In[33]:


img = cv2.imread('images/blurring.png')
imshow('Image', img)


# ## Average blur

# We are going to define a *k* x *k* sliding window or kernel, where *k* is always an **odd number**. We will then take the average of all the surrounding pixels in the *k* x *k* neighbour hood and assign it to the centre pixel.

# In[34]:


blurred = cv2.blur(img, (7,7))
imshow('Blurred Image', blurred)


# `cv2.blur`, the first parameter is the source image on which were we have to apply average blurring, the second parameter is a tuple specifying the size of the kernel.

# ## Gaussian blur

# Gaussian blurring is similar to average blurring, but instead of using a simple mean, we are now using a weighted mean, where neighborhood pixels that are closer to the central pixel contribute more “weight” to the average.
# 
# The end result is that our image is more naturally blurred, than using the average method discussed previously.

# In[35]:


blurred = cv2.GaussianBlur(img, (11,11),0)
imshow('Blurred Image', blurred)


# `cv2.GaussianBlur`, to which we have provided three parameter first one being source image, second is the kernel size and the third is value of sigma in gaussian function (don't worry to much about it, just keep it 0). 

# ## Median blur

# Median Blur is similar to average blur instead of taking average we are going to take median of every pixel in the filter and the value is going to be the value of center pixel.

# In[36]:


blurred = cv2.medianBlur(img,5)
imshow('Blurred Image', blurred)


# `cv2.medianBlur`, the first parameter is source image and the second parameter is kernel size. With median blur we got really good result, it looks even better than the original image.
# 
# **Note:** While other filters might be often useful, this method is highly effective in removing salt-and-pepper noise. A larger kernel size will use a larger matrix, and take pixels from further away from the center, which may or may not help. Whether you want a larger or smaller kernel really depends on the image, but 5 is a good number to start with.

# ## Bilateral filtering

# All the above methods tend to blur edges. This is not the case for the bilateral filter, `cv2.bilateralFilter`, it is highly effective at noise removal while preserving edges. But the operation is slower compared to other filters.

# In[37]:


blurred = cv2.bilateralFilter(img, 9, 25, 25)
imshow('Blurred Image', blurred)


# `cv2.bilateralFilter`, the first parameter is source image, the second parameter is diameter of each pixel neighborhood. The next two arguments are for *filter sigma in the color space* and *filter sigma in the coordinate space*. You can read more about the arguments, [here](https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter). Still, the best way would be to try changing them and look at the output image.

# Lets compare the output of all these smoothing techniques, together! 

# In[38]:


avg_blur = cv2.blur(img, (7,7))
cv2.imshow('Average Blur', avg_blur)

gas_blur = cv2.GaussianBlur(img, (11,11),0)
cv2.imshow('Gaussian Blur', gas_blur)

med_blur = cv2.medianBlur(img,5)
cv2.imshow('Median Blur', med_blur)

bil_blur = cv2.bilateralFilter(img, 9, 25, 25)
cv2.imshow('Bilateral Blur', bil_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()


# The choice of smoothing technique is very application dependent. If preserving edges is important to you, then use bilateral filtering. If your image has salt-and-pepper noise then use median filtering. 
# 
# Generally speaking, gaussian filtering is a good starting point. Try playing with the kernel size until you get the desired results. Use other techniques only if you know what you are doing or the application demands it.
# 
# **Key Takeaways**
# 
# Applying a blur to an image smooths edges and removes noise from the image. Blurring is often used as a first step before we perform Thresholding, Edge Detection, or before we find the Contours of an image. Larger kernel size may remove more noise, but they will also remove detail from an image.

# ## Application time

# There are many more smoothing techniques and algorithms in OpenCV, but we cannot learn about them all. Many of them are very application specific and you will hardly need them. 
# 
# **Note:** There are implementations for almost 2500 algorithms in OpenCV. Nobody can learn all of them. Also, nobody need all these algorithms together. So, when learning OpenCV, you should  focus on learning *how to use a particular function ?* and don't worry about the implementation.  

# ### Oil painting

# One very interesting smoothing algorithm that makes your image look like its oil painted !

# In[39]:


img = cv2.imread('images/friends2.jpeg')
img = cv2.xphoto.oilPainting(img, 3, 1)
imshow('Image', img)


# Amazing, right? OpenCV can make you an artist in real life. 
# 
# **Exercise:** 
# - Try using a different image. Also, play with the values of the arguments to understand there impact better.

# ### Water color

# You guessed it, this technique makes your image look like water colored 

# In[40]:


img = cv2.imread('images/friends2.jpeg')
img = cv2.edgePreservingFilter(img, flags=1, sigma_s=160, sigma_r=0.7)
imshow('Image', img)


# You are on a roll! Its time that you showcase your artistic skill to your friends, OpenCV got you covered.
# 
# **Exercise:** 
# - Try using a different image. Also, play with the values of *sigma_s* and *sigma_r* to understand them better. Don't change the value of *flags*.

# ### Pencil Sketch

# Lets try to make a pencil sketch from an image.

# In[41]:


img = cv2.imread('images/friends2.jpeg')

# pencil sketch code
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_invert = cv2.bitwise_not(img_gray)
img_smoothing = cv2.GaussianBlur(img_invert, (91, 91), sigmaX=0, sigmaY=0)
final_img = cv2.divide(img_gray, 255 - img_smoothing, scale=256)

imshow('Image', final_img)


# You have a pencil sketch in less than a second, pretty dope! Lets try to understand it.
# 
# - **Step - 1:** convert our image to gray-scale
# - **Step - 2:** invert the black and white colors in the gray-scale image from *step-1*
# - **Step - 3:** take the inverted image from *step-2* and apply gaussian blur to it
# - **Step - 4:** perform `(img_gray * 256)/(255 - img_smoothing)` using `cv2.divide`
# 
# You already know `cv2.cvtColor` and `cv2.GaussianBlur` functions. There are only two new functions here: `cv2.divide` and `cv2.bitwise_not`. They are called *arithematic operations* and *bitwise operations* respectively. We will explore them next.

# ## Arithematic operations

# We all know basic arithmetic operations like addition and subtraction. But when working with images, we need to keep in mind the limits of our color space and data type.
# For example, RGB images have pixels that fall within the range [0, 255]. So what happens if we are examining a pixel with intensity 250 and we try to add 10 to it? Under normal arithmetic rules, we would end up with a value of 260. However, since RGB images are represented as 8-bit unsigned integers, 260 is not a valid value.
# 
# So, what should happen? Should we perform a check of some sort to ensure no pixel falls outside the range of [0, 255], thus clipping all pixels to have a minimum value of
# 0 and a maximum value of 255? Or do we apply a modulus operation, and “wrap around”? Under modulus rules, adding 10 to 250 would simply wrap around to a value of 4.
# 
# Which way is the “correct” way to handle image additions and subtractions that fall outside the range of [0, 255]? The answer is there is no correct way – it simply depends on how you are manipulate your pixels and what you want the desired results to be. However, be sure to keep in mind that there is a difference between OpenCV and NumPy addition. NumPy will perform modulo arithmetic and “wrap around”. OpenCV, on the other hand, will perform clipping and ensure pixel values never fall outside the range [0, 255]. 
# 
# But don’t worry! These nuances will become clearer as we explore some code.

# In[42]:


a = np.ones((5, 5, 3), dtype='uint8') * 100
b = np.ones((5, 5, 3), dtype='uint8') * 200


# In[43]:


# OpenCV arithematics
max_cv = cv2.add(a, b)
min_cv = cv2.subtract(a, b)

max_cv.max(), min_cv.min()


# In[44]:


# Numpy arithematics
max_np = a + b
min_np = a - b

max_np.max(), min_np.min()


# Remember the difference we mentioned between OpenCV and NumPy addition above? Well, ponder for a minute to ensure you fully understand it.
# 
# OpenCV takes care of clipping for us, and ensures that the arithematic operations produces a maximum value of *255* and a minimum value of *0*. On the other hand, NumPy does not perform clipping – it instead performs modulo arithmetic and “wraps around”. Once a value of 255 is reached, NumPy wraps around to zero, and then starts counting up again. And if it reaches 0, then it wraps around 255 and starts counting down.
# 
# When performing integer arithmetic, it is important to keep in mind your desired output. Do you want all values to be clipped if they fall outside the range [0, 255]? Then use OpenCV’s built-in methods for image arithmetic.
# 
# Do you want modulus arithmetic operations and have values wrap around if they fall outside the range of [0, 255]? Then simply add and subtract the NumPy arrays as you
# normally would.
# 
# **Note:** For all arithematic operations between arrays, the input arrays *must have the same size*.
# 
# Now that we have explored the caveats of image arithmetic in OpenCV and NumPy, let’s perform the arithmetic on actual images and view the results:

# In[45]:


img = cv2.imread('images/friends2.jpeg')
a = np.ones(img.shape, dtype='uint8') * 100

added = cv2.add(img, a)
cv2.imshow('Added', added)

subtracted = cv2.subtract(img, a)
cv2.imshow('Subtracted', subtracted)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Once you have learned arithematic operation using OpenCV, now do you understand what this line of code is doing in the pencil sketch code:
# 
# ```python
# final_img = cv2.divide(img_gray, 255 - img_smoothing, scale=256)
# ```
# 
# It is simply taking two images (i.e. numpy array) and dividing them element-wise. After the division operation, we multiple the resultant array by 256 to bring everything back to [0-255] range. 

# In this section, we explored the peculiarities of image arithmetic using OpenCV and NumPy. These caveats are important to keep in mind, otherwise you may get unwanted results when performing arithmetic operations on your images.

# ## Bitwise operations

# They are very similar to AND, OR, XOR and NOT operations that we learned in our college. The best way to learn about bitwise operation is to use the classic example of square and circle mask.
# 
# **Note:** if you don't remember the truth table for any operation then please refer the table below.
# 
# 
# | p | q | p AND q | p OR q | NOT p | p XOR q |
# |---|---|---------|--------|-------|---------|
# | T | T | T       | T      | F     | F       |
# | T | F | F       | T      | F     | T       |
# | F | T | F       | T      | T     | T       |
# | F | F | F       | F      | T     | F       |
# 
# 
# Lets write some code, shall we?

# In[46]:


base_canvas = np.zeros((300, 300), dtype = "uint8")
rectangle = cv2.rectangle(base_canvas, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)

base_canvas = np.zeros((300, 300), dtype = "uint8")
circle = cv2.circle(base_canvas, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)

cv2.waitKey(0)
cv2.destroyAllWindows()


# We just created two images: one with a fill-in rectangle and other with a fill-in circle. 
# 
# In order to utilize bitwise functions, we assume (in most cases) that we are comparing two pixels (the only exception is the NOT function). Hence, its important that input images have same shape, else you will get an error complaining about it. We’ll compare each of the pixels and then construct our bitwise representation. 
# 
# Let’s quickly review our binary operations:
# - **AND:** A bitwise AND is true if and only if both pixels are greater than zero.
# - **OR:** A bitwise OR is true if either of the two pixels are greater than zero.
# - **XOR:** A bitwise XOR is true if and only if either of the two pixels are greater than zero, but not both.
# - **NOT:** A bitwise NOT inverts the “on” and “off” pixels in an image.
# 

# In[47]:


base_canvas = np.zeros((300, 300), dtype = "uint8")
circle = cv2.circle(base_canvas, (150, 150), 150, 255, -1)

base_canvas = np.zeros((300, 300), dtype = "uint8")
rectangle = cv2.rectangle(base_canvas, (25, 25), (275, 275), 255, -1)

bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)

bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)

bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)

bitwiseNot = cv2.bitwise_not(rectangle)
cv2.imshow("NOT", bitwiseNot)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Can you see the output of each operation? Does it all make sense ? Spend a little time to collect your thoughts.
# 
# Can you understand this line now?
# 
# ```python
# img_invert = cv2.bitwise_not(img_gray)
# ```
# It simply invert our gray-scale image. We would really recommend you to visit the pencil sketch code again and see if you can understand it, now that you know ever function that is used.
# 
# Overall, bitwise operations are extremely simple, yet very powerful. And they are absolutely essentially important in computer vision. But you will say, what are they useful for? Other than inverting gray-scale images, they are many other applications were bitwise operations are used. One such application is masking. We will learn about other applications later in the course. For now, lets talk about masking.

# ## Masking
# 
# Using a mask allows us to focus only on the portions of the image that interests us.

# In[48]:


mask = np.zeros(img.shape[:2], dtype='uint8')
mask = cv2.rectangle(mask, (222, 212), (378, 358), 255, -1)
cv2.imshow("Mask", mask)

masked = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Masked", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()


# We construct a NumPy array, filled with zeros, with the same width and height as our image. We then draw our white rectangle.
# 
# We apply our mask using the `cv2.bitwise_and` function. The first two parameters are the image itself. Obviously, the AND function will be True for all pixels in the image; however, the important part of this function is the *mask* keyword argument. By supplying a mask, the `cv2.bitwise_and` function only examines pixels that are “on” in the mask. In this case, only pixels that are part of the white rectangle.
# 
# **Exercise:** 
# - Try making a circular mask around monica's face. You already know how to detect faces. Once you have the cordinates of the box, calculate the its center. 

# We learned many new concepts in this notebook. The best way to make sure you understand all of them, is to make build some applications using them.
# 
# **Excercise:**
# - Refer the previous notebook and try to develop a real-time sketch application. Its should take input images from the webcam and make sketch images in real-time. 
# - Try to extend the real-time sketch app to generate oil painting and water color versions of videos from the webcam
# - Apply a circular mask to video feed from the webcam.  

# ## Questionaire
# 
# - Smoothing and blurring?
# - When to use median filtering?
# - When to use bilateral filtering?
# - Which one clips the values after 255, Numpy or OpenCV?
# - What will you get if you do `244 + 16`? First using numpy and then using OpenCV arithematics.
# - Bitwise AND, OR, XOR and NOT?
# - Masking ?
# 
# In the next notebook, we will learn about thresholding and edge detection.
