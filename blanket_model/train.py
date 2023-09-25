import os
import cv2
import argparse
from sklearn.svm import LinearSVC
from skimage import feature, exposure, color
from skimage.feature import hog
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

    return fd, hog_image_rescaled

x = 550
y = 250
h = 700
w = 550
frame_dim = (1920,1080)
fps = 20
def hog_the_video(input_path):
    cap = cv2.VideoCapture(input_path)
    instance_count = 0
    next_frame = 0
    count = 0
    success = True
    video_fds = []
    video_images = []
    while success:
        success, frame = cap.read()
        if count % fps == 0 and success:
            if frame.shape[0] > 1080 and frame.shape[1] > 1920:
                frame = maintain_aspect_ratio_resize(frame.copy(), width=frame_dim[0], height=frame_dim[1])

            cropped_frame = frame[y:y+h, x:x+w]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            ax1.axis('off')
            ax1.imshow(cropped_frame, cmap=plt.cm.gray)
            ax1.set_title('Input image')

            cropped_frame_forced_size = cv2.resize(cropped_frame, (512, 512))
            ax2.axis('off')
            ax2.imshow(cropped_frame_forced_size, cmap=plt.cm.gray)
            ax2.set_title('Histogram of Oriented Gradients')
            plt.show()

            # TODO: test impacts of gaus blur
            blur = cv2.GaussianBlur(cropped_frame,(5,5),0)

            fd, hoggy_cropped_frame = hog_it_up(blur)
            print('shape: ', hoggy_cropped_frame.shape)

            video_fds.append(fd)
            video_images.append(hoggy_cropped_frame)

            # file_name = input_path.rsplit('\\', 1)[-1] + '_frame' + str(instance_count) + '.jpg'
            # name = f'{output_path}\\{file_name}'
            # print ('Creating...' + name)
            # cv2.imwrite(name, hoggy_cropped_frame)
            # plt.imsave(name, hoggy_cropped_frame)
            # instance_count += 1
        count+=1
    
    cap.release()
    cv2.destroyAllWindows()
    return video_fds, video_images

### iterate over raw video inputs, select images throughout video, get the hog feature descriptors & images 
all_images = []
all_fds = []
all_labels = []
video_paths = os.listdir(f"./raw")
for path in video_paths:
    all_videos = os.listdir(f"raw/{path}")
    for video in all_videos:
        video_path = f"./raw/{path}/{video}"
        print('video_path: ', video_path)

        fds, images = hog_the_video(video_path)

        # image = cv2.resize(image, (128, 256))
        # get the HOG descriptor for the image
        # hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
        #     cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    
        print(path)
        print(images)
        all_images = all_images + images
        all_fds = all_fds + fds
        all_labels = all_labels + [path] * len(fds)
