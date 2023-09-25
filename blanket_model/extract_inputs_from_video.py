import cv2
import os
import sys
import time
from skimage.feature import hog
from skimage import exposure
from skimage import color
import matplotlib.pyplot as plt
from PIL import Image

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

input_path = sys.argv[1]
output_path = sys.argv[2]

def hog_it_up(image):
    fd, hog_image = hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), channel_axis=None, visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')
    # # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

    return hog_image_rescaled

x = 550
y = 250
h = 700
w = 550

frame_dim = (1920,1080)
cap = cv2.VideoCapture(input_path)
instance_count = 0
fps = 60
next_frame = 0
count = 0
counter = 1
success = True
while success:
    success, frame = cap.read()
    if count % fps == 0:
        if frame.shape[0] > 1080 and frame.shape[1] > 1920:
            frame = maintain_aspect_ratio_resize(frame.copy(), width=frame_dim[0], height=frame_dim[1])
    
        cropped_frame = frame[y:y+h, x:x+w]

    # TODO: test impacts of gaus blur
        blur = cv2.GaussianBlur(cropped_frame,(5,5),0)

        hoggy_cropped_frame = hog_it_up(blur)

        file_name = input_path.rsplit('\\', 1)[-1] + '_frame' + str(instance_count) + '.jpg'

        name = f'{output_path}\\{file_name}'
        print ('Creating...' + name)

        # cv2.imwrite(name, hoggy_cropped_frame)
        plt.imsave(name, hoggy_cropped_frame)

        instance_count += 1
    count+=1
  
cap.release()
cv2.destroyAllWindows()