'''
# https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/#:~:text=Morphological%20operations%20are%20simple%20transformations,as%20well%20as%20decrease%20them.
'''
#%%
import cv2
import matplotlib.pyplot as plt

# %%
cv2.__version__
# %%
args = {'img1':r'../data/DJI_0027.png'}
# load the image, convert it ot grayscale, display it
image = cv2.imread(args["img1"])
print(type(image))
image.shape
# %%
fig = plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()
# %%
fig, axes = plt.subplots(3,1,figsize=(10,24))
# apply a series of erosions
for i in range(0,3):
    eroded = cv2.erode(image.copy(), None,iterations=i+1)
    axes[i].imshow(eroded)
    axes[i].set_title('Eroded {} times'.format(i+1))
plt.show()
# %%
# apply a series of dilations
# structing element is set to None, which means 3X3 8 neighboring
fig, axes = plt.subplots(3,1,figsize=(10,24))
for i in range(0,3):
    eroded = cv2.dilate(image.copy(), None,iterations=i+1)
    axes[i].imshow(eroded)
    axes[i].set_title('Dilated {} times'.format(i+1))
plt.show()
# %%
# An opening operation is an erosion followed by a dialtion
fig, axes = plt.subplots(3,1,figsize=(10,24))
kernelSizes = [(3,3), (5,5), (7,7)]
for i, kernelSize in enumerate(kernelSizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    axes[i].imshow(opening)
    axes[i].set_title('Opening: ({}, {})'.format(kernelSize[0], kernelSize[1]))
plt.show()
# %%
# A closing operation is an dialation followed by an erosion
fig, axes = plt.subplots(3,1,figsize=(10,24))
kernelSizes = [(3,3), (5,5), (7,7)]
for i, kernelSize in enumerate(kernelSizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    axes[i].imshow(closing)
    axes[i].set_title('closing: ({}, {})'.format(kernelSize[0], kernelSize[1]))
plt.show()
# %%
# Morphological gradient
'''
it is the difference between a dilation and erosion.
it is useful to determining the outline of a particular object.
'''
fig, axes = plt.subplots(3,1,figsize=(10,24))
kernelSizes = [(3,3), (5,5), (7,7)]
for i, kernelSize in enumerate(kernelSizes):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernelSize)
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    axes[i].imshow(gradient)
    axes[i].set_title('gradient: ({}, {})'.format(kernelSize[0], kernelSize[1]))
plt.show()
# %% 
# Top hat and black hat
# Top hat also know as white hat. it is the difference
# between the original and the opening
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
blackHat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, rectKernel)
topHat   = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rectKernel)
fig, axes = plt.subplots(2,1,figsize=(10,16))
axes[0].imshow(blackHat)
axes[0].set_title('blackHat')
axes[1].imshow(topHat)
axes[1].set_title('topHat')
plt.show()