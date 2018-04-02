def pre_processing(kaggle_folder_path):
    import numpy as np
    import pathlib
    import cv2
    import os

    #This section will define all the starting paths that will be needed
    stage1_train_path = os.path.join(kaggle_folder_path, "stage1_train")
    stage1_test_path = os.path.join(kaggle_folder_path, "stage1_test")
    #This builds paths to the augmented train data folders we want to create
    train_data_augmentation_path = os.path.join(kaggle_folder_path, "train_data_augmentation")
    image_train_augmentation_path = os.path.join(train_data_augmentation_path, "image_train_augmentation")
    mask_train_augmentation_path = os.path.join(train_data_augmentation_path, "mask_train_augmentation")
    #This builds paths to the augmented test data folders we want to create
    test_data_augmentation_path = os.path.join(kaggle_folder_path, "test_data_augmentation")
    image_test_augmentation_path = os.path.join(test_data_augmentation_path, "image_test_augmentation")
    #This part does the actual building of the folders
    if not os.path.exists(train_data_augmentation_path):
        os.makedirs(train_data_augmentation_path)
    if not os.path.exists(image_train_augmentation_path):
        os.makedirs(image_train_augmentation_path)
    if not os.path.exists(mask_train_augmentation_path):
        os.makedirs(mask_train_augmentation_path)
    if not os.path.exists(test_data_augmentation_path):
       os.makedirs(test_data_augmentation_path) 
    if not os.path.exists(image_test_augmentation_path):
        os.makedirs(image_test_augmentation_path)
    
    #Now start building the train folders up
    for folder_train in os.listdir(stage1_train_path):
        #os.listdir(path) will list the items in the directory
        stage1_train_folder_path = "%s" % folder_train
        stage1_train_folder_path = os.path.join(stage1_train_path, stage1_train_folder_path)
        image_and_masks_train_path = os.path.join(stage1_train_folder_path, "image_and_masks")
        if not os.path.exists(image_and_masks_train_path):
            os.makedirs(image_and_masks_train_path)
        for items_train in os.listdir(stage1_train_folder_path): 
            if items_train == "images":
                stage1_train_folder_images_path ="%s" % items_train
                stage1_train_folder_images_path = os.path.join(stage1_train_folder_path, stage1_train_folder_images_path)        
                for images_train in os.listdir(stage1_train_folder_images_path):
                    stage1_train_images_list_path = "%s" % images_train
                    stage1_train_images_list_path = os.path.join(stage1_train_folder_images_path, stage1_train_images_list_path)
                    original_img_train = cv2.imread(stage1_train_images_list_path,1)
                    height_train,width_train,channels_train = np.shape(original_img_train)
                move_image = os.path.join(image_and_masks_train_path, "image.png")
                cv2.imwrite(move_image, original_img_train)    
                                   
            if items_train == "masks":
                all_masks_train = np.zeros([height_train,width_train,channels_train],np.uint8)
                #this creates a blank image to start with for the adding of masks
                stage1_train_folder_mask_path = "%s" % items_train
                stage1_train_folder_mask_path = os.path.join(stage1_train_folder_path, stage1_train_folder_mask_path)
                for masks_train in os.listdir(stage1_train_folder_mask_path):
                    stage1_train_masks_list_path = "%s" % masks_train
                    stage1_train_masks_list_path = os.path.join(stage1_train_folder_mask_path, stage1_train_masks_list_path)
                    single_train_mask = cv2.imread(stage1_train_masks_list_path,1)
                    all_masks_train = cv2.add(all_masks_train, single_train_mask)
                combined_masks_path = os.path.join(image_and_masks_train_path, "combined_masks.png")
                #this will put all_masks_train image and save it into the file called combined_masks
                cv2.imwrite(combined_masks_path, all_masks_train)
    
    #Now start building the test folders up
    for folder_test in os.listdir(stage1_test_path):
        #os.listdir(path) will list the items in the directory
        stage1_test_folder_path = "%s" % folder_test
        stage1_test_folder_path = os.path.join(stage1_test_path, stage1_test_folder_path)
        image_and_masks_test_path = os.path.join(stage1_test_folder_path, "image_and_masks")
        if not os.path.exists(image_and_masks_test_path):
            os.makedirs(image_and_masks_test_path)
        for items_test in os.listdir(stage1_test_folder_path): 
            if items_test == "images":
                stage1_test_folder_images_path ="%s" % items_test
                stage1_test_folder_images_path = os.path.join(stage1_test_folder_path, stage1_test_folder_images_path)        
                for images_test in os.listdir(stage1_test_folder_images_path):
                    stage1_test_images_list_path = "%s" % images_test
                    stage1_test_images_list_path = os.path.join(stage1_test_folder_images_path, stage1_test_images_list_path)
                    original_img_test = cv2.imread(stage1_test_images_list_path,1)
                    height_test,width_test,channels_test = np.shape(original_img_test)
                move_image = os.path.join(image_and_masks_test_path, "image.png")
                cv2.imwrite(move_image, original_img_test)    
                                   
            if items_test == "masks":
                all_masks_test = np.zeros([height_test,width_test,channels_test],np.uint8)
                #this creates a blank image to start with for the adding of masks
                stage1_test_folder_mask_path = "%s" % items_test
                stage1_test_folder_mask_path = os.path.join(stage1_test_folder_path, stage1_test_folder_mask_path)
                for masks_test in os.listdir(stage1_test_folder_mask_path):
                    stage1_test_masks_list_path = "%s" % masks_test
                    stage1_test_masks_list_path = os.path.join(stage1_test_folder_mask_path, stage1_test_masks_list_path)
                    single_test_mask = cv2.imread(stage1_test_masks_list_path,1)
                    all_masks_test = cv2.add(all_masks_test, single_test_mask)
                combined_masks_path = os.path.join(image_and_masks_test_path, "combined_masks.png")
                #this will put all_masks_test image and save it into the file called combined_masks
                cv2.imwrite(combined_masks_path, all_masks_test)


def make_data_tensor(kaggle_folder_path):    
    #type in terminal to remove DS.Store
    # sudo find / -name ".DS_Store" -depth -exec rm {} \;
    import os
    import cv2

    image_tensor = []
    mask_tensor = []

    stage_1_train_path = os.path.join(kaggle_folder_path, "stage1_train")
    #stage_1_train_path = re.sub(".DS_Store", "", stage_1_train_path)
    print(stage_1_train_path)

    for folder in os.listdir(stage_1_train_path):
        images_and_masks_folder_path = "%s" % folder
        images_and_masks_folder_path = os.path.join(stage_1_train_path, images_and_masks_folder_path, "image_and_masks")
        print(images_and_masks_folder_path)
        for file in os.listdir(images_and_masks_folder_path):
            if file == "combined_masks.png":
                mask_path = "%s" % file
                mask_path = os.path.join(images_and_masks_folder_path, mask_path)
                print(mask_path)
                mask_img = cv2.imread(mask_path,1)         
                resized_mask = cv2.resize(mask_img, (256,256))
                mask_tensor.append(resized_mask)
            else:
                image_path = os.path.join(images_and_masks_folder_path, file)
                print(image_path)
                image_img = cv2.imread(image_path,1)         
                resized_image = cv2.resize(image_img, (256,256))
                image_tensor.append(resized_image)  
    return image_tensor, mask_tensor 


    def kaggle_data_augmentation(training_image_tensor, training_mask_tensor):
    
    import tensorflow as tf
    import numpy as np
    import time
    import progressbar
    from keras.preprocessing.image import random_shear, random_zoom

    augmented_image_tensor = []
    augmented_mask_tensor = []
    model = tf.global_variables_initializer()
    
    with tf.Session() as session:
        with progressbar.ProgressBar(max_value=len(training_image_tensor)) as bar:
            for number in range(1,len(training_image_tensor)):
                #part where we augment the original images
                #to  keep indexing cosistent will group image and mask togther
                #rather than all images then all the masks.
                image = training_image_tensor[number][:][:][:]
                mask = training_mask_tensor[number][:][:][:]
                imageT = tf.transpose(image, perm=[1, 0, 2])
                maskT = tf.transpose(mask, perm=[1, 0, 2])
                imageFUD = tf.image.flip_up_down(image)
                maskFUD = tf.image.flip_up_down(mask)
                imageR90 = tf.image.rot90(image,k=1,name=None)
                maskR90 = tf.image.rot90(mask,k=1,name=None)
                imageR270 = tf.image.rot90(image,k=3,name=None)
                maskR270 = tf.image.rot90(mask,k=3,name=None)
                imageAB = tf.image.adjust_brightness(image, delta=.5)
                maskAB = tf.image.adjust_brightness(mask, delta=.5)
                imageAC = tf.image.adjust_contrast(image, contrast_factor=.5)
                maskAC = tf.image.adjust_contrast(mask, contrast_factor=.5)
                imageAH = tf.image.adjust_hue(image, .5, name=None)
                maskAH = tf.image.adjust_hue(mask, .5, name=None)
                session.run(model)

                #images_file_path = "%s" % number
                #images_file_path = os.path.join(stage1_train_folder_images_path,images_file_path)
                augmented_image_tensor.append(image)
                imageT = session.run(imageT)
                augmented_image_tensor.append(imageT)
                imageFUD = session.run(imageFUD)
                augmented_image_tensor.append(imageFUD)
                imageR90 = session.run(imageR90)
                augmented_image_tensor.append(imageR90)
                imageR270 = session.run(imageR270)
                augmented_image_tensor.append(imageR270)
                imageAB = session.run(imageAB)
                augmented_image_tensor.append(imageAB)
                imageAC = session.run(imageAC)
                augmented_image_tensor.append(imageAC)
                imageAH = session.run(imageAH)
                augmented_image_tensor.append(imageAH)
                imageRS = random_shear(image, 2, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                augmented_image_tensor.append(imageRS)
                imageRZ = random_zoom(image, [.9,.8], row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                augmented_image_tensor.append(imageRZ)

                augmented_mask_tensor.append(mask)    
                maskT = session.run(maskT)
                augmented_mask_tensor.append(maskT)
                maskFUD = session.run(maskFUD)
                augmented_mask_tensor.append(maskFUD)
                maskR90 = session.run(maskR90)
                augmented_mask_tensor.append(maskR90)
                maskR270 = session.run(maskR270)
                augmented_mask_tensor.append(maskR270)
                maskAB = session.run(maskAB)
                augmented_mask_tensor.append(maskAB)
                maskAC = session.run(maskAC)
                augmented_mask_tensor.append(maskAC)
                maskAH = session.run(maskAH)
                augmented_mask_tensor.append(maskAH)
                maskRS = random_shear(mask, 2, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                augmented_mask_tensor.append(maskRS)
                maskRZ = random_zoom(mask, [.9,.8], row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                augmented_mask_tensor.append(maskRZ)
                bar.update(number)
    return  augmented_image_tensor, augmented_mask_tensor  


###---------Start of the model------------###

import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, UpSampling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, Add

img_rows = 128
img_cols = 128
inputs = Input((img_rows, img_cols,3))
x_train = np.random.random((100, 128, 128, 3))
y_train = np.random.random((100, 128, 128, 1))
x_test = np.random.random((20, 100, 100, 3))
y_test = np.random.random((20, 100, 100, 1))

conv1_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_0)
conv1_2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

conv2_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_0)
conv2_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

conv3_0 = Conv2D(256, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3_1 = Conv2D(256, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_0)
conv3_2 = Conv2D(256, 4, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

conv4_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_0)
drop4 = Dropout(0.5)(conv4_1)
pool4 = MaxPooling2D(pool_size=(2, 2),padding = 'same')(drop4)

conv5_0 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5_1 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_0)
drop5 = Dropout(0.5)(conv5_1)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = Add()([drop4,up6])
conv6_0 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_0)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_1))
merge7 = Add()([conv3_1,up7])
conv7_0 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_0)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_1))
merge8 = Add()([conv2_1,up8])
conv8_0 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8_1 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_0)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_1))
merge9 = Add()([conv1_1,up9])
conv9_0 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9_1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_0)
conv9_2 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9_1)

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9_2)

model = Model(input = inputs, output = conv10)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=10)
