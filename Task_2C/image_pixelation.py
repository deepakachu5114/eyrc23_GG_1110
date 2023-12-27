#import those 2 modules I mentioned

import os
import shutil
from PIL import Image
import numpy as np

# shutil.copytree("/home/deepakachu/Desktop/eyantra/task2b/training", "/home/deepakachu/Desktop/eyantra/task2b/training_resized_2b")
# shutil.copytree("/home/deepakachu/Desktop/eyantra/task2b/validation", "/home/deepakachu/Desktop/eyantra/task2b/validation_resized_2b")
# shutil.copytree("/home/deepakachu/Desktop/eyantra/task2b/testing", "/home/deepakachu/Desktop/eyantra/task2b/testing_resized_2b")

for dirs in os.listdir("/home/deepakachu/Desktop/eyantra/task2b/validation_resized_2b"):
    for files in os.listdir(os.path.join("/home/deepakachu/Desktop/eyantra/task2b/validation_resized_2b",dirs)):
        im = Image.open(os.path.join("/home/deepakachu/Desktop/eyantra/task2b/validation_resized_2b", dirs, files))

        # w, h = im.size
        # # find NEW dimensions from user-defined number (50% for example)
        # new_w = w * 0.2
        # new_h = h * 0.2
        # # round to nearest whole number and convert from float to int
        # new_w = np.round(new_w)
        # new_w = int(new_w)
        # new_h = np.round(new_h)
        # new_h = int(new_h)
        # # downsample image to these new dimensions
        # down_sampled = im.resize((new_w, new_h))
        print("ori ", im.size)
        # upsample back to original size (using "4" to signify bicubic)
        resized_image = im.resize((224,224), resample=Image.BICUBIC)
        print("res ",resized_image.size)# save the image
        resized_image.save(os.path.join("/home/deepakachu/Desktop/eyantra/task2b/validation_resized_2b", dirs, files))

#open desired image
# im = Image.open('/home/deepakachu/Desktop/eyantra/task_2c/sample.png')
# #find its width & height
# w,h = im.size
# #find NEW dimensions from user-defined number (50% for example)
# new_w = w * 0.2
# new_h = h * 0.2
# #round to nearest whole number and convert from float to int
# new_w = np.round(new_w)
# new_w = int(new_w)
# new_h = np.round(new_h)
# new_h = int(new_h)
# #downsample image to these new dimensions
# down_sampled = im.resize((new_w, new_h))
# #upsample back to original size (using "4" to signify bicubic)
# up_sampled = down_sampled.resize((w, h), resample = 4)
# #save the image
# up_sampled.save('pixelated.png')