# %%
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
# %%
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str,default='../data/DJI_0027.JPG', 
                help='Path to the input image')
args = vars(ap.parse_args())
print(args)
# %%
image = cv2.imread(args['image'])
print(image.shape)
#cv2.imshow("original", image)
#cv2.waitKey(0)
fig = plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()
