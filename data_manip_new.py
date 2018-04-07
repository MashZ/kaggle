def kaggle_data_augmentation(training_image_tensor, training_mask_tensor, kaggle_folder_path):
    
    import tensorflow as tf
    import numpy as np
    import time
    import progressbar
    import cv2
    import os
    #import keras
    #from keras.preprocessing.image import random_shear, random_zoom

    training_data_augmentation_path = os.path.join(kaggle_folder_path, "training_data_augmentation")
    image_train_augmentation_path = os.path.join(training_data_augmentation_path, "image_train_augmentation")
    mask_train_augmentation_path = os.path.join(training_data_augmentation_path, "mask_train_augmentation")

    augmented_image_tensor = []
    augmented_mask_tensor = []
    model = tf.global_variables_initializer()
    
    with tf.Session() as session:
        with progressbar.ProgressBar(max_value=len(training_image_tensor)) as bar:
            image_counter = 0
            mask_counter = 0
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

                #start the image augmentation
                #augmented_image_tensor.append(image)
                image_counter = 1 + image_counter
                image_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(image_path, image)
                
                imageT = session.run(imageT)
                #augmented_image_tensor.append(imageT)
                image_counter = 1 + image_counter
                imageT_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageT_path, imageT)
                
                imageFUD = session.run(imageFUD)
                #augmented_image_tensor.append(imageFUD)
                image_counter = 1 + image_counter
                imageFUD_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageFUD_path, imageFUD)

                imageR90 = session.run(imageR90)
                #augmented_image_tensor.append(imageR90)
                image_counter = 1 + image_counter
                imageR90_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageR90_path, imageR90)

                imageR270 = session.run(imageR270)
                #augmented_image_tensor.append(imageR270)
                image_counter = 1 + image_counter
                imageR270_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageR270_path, imageR270)

                imageAB = session.run(imageAB)
                #augmented_image_tensor.append(imageAB)
                image_counter = 1 + image_counter
                imageAB_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageAB_path, imageAB)    

                imageAC = session.run(imageAC)
                #augmented_image_tensor.append(imageAC)
                image_counter = 1 + image_counter
                imageAC_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageAC_path, imageAC)

                imageAH = session.run(imageAH)
                #augmented_image_tensor.append(imageAH)
                image_counter = 1 + image_counter
                imageAH_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageAH_path, imageAH)

                imageRS = random_shear(image, 2, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                #augmented_image_tensor.append(imageRS)
                image_counter = 1 + image_counter
                imageRS_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageRS_path, imageRS)

                imageRZ = random_zoom(image, [.9,.8], row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                #augmented_image_tensor.append(imageRZ)
                image_counter = 1 + image_counter
                imageRZ_path = os.path.join(image_train_augmentation_path, "test_image_" + "%s" %image_counter + ".png")
                cv2.imwrite(imageRZ_path, imageRZ)

                #start the mask augmentation
                #augmented_mask_tensor.append(mask)
                mask_counter = 1 + mask_counter
                mask_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(mask_path, mask)

                maskT = session.run(maskT)
                #augmented_mask_tensor.append(maskT)
                mask_counter = 1 + mask_counter
                maskT_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskT_path, maskT)

                maskFUD = session.run(maskFUD)
                #augmented_mask_tensor.append(maskFUD)
                mask_counter = 1 + mask_counter
                maskFUD_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskFUD_path, maskFUD)

                maskR90 = session.run(maskR90)
                #augmented_mask_tensor.append(maskR90)
                mask_counter = 1 + mask_counter
                maskR90_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskR90_path, maskR90)

                maskR270 = session.run(maskR270)
                #augmented_mask_tensor.append(maskR270)
                mask_counter = 1 + mask_counter
                maskR270_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskR270_path, maskR270)

                maskAB = session.run(maskAB)
                #augmented_mask_tensor.append(maskAB)
                mask_counter = 1 + mask_counter
                maskAB_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskAB_path, maskAB)

                maskAC = session.run(maskAC)
                #augmented_mask_tensor.append(maskAC)
                mask_counter = 1 + mask_counter
                maskAC_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskAC_path, maskAC)

                maskAH = session.run(maskAH)
                #augmented_mask_tensor.append(maskAH)
                mask_counter = 1 + mask_counter
                maskAH_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskAH_path, maskAH)

                maskRS = random_shear(mask, 2, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                #augmented_mask_tensor.append(maskRS)
                mask_counter = 1 + mask_counter
                maskRS_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskRS_path, maskRS)

                maskRZ = random_zoom(mask, [.9,.8], row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest', cval=0.0)
                #augmented_mask_tensor.append(maskRZ)
                mask_counter = 1 + mask_counter
                maskRZ_path = os.path.join(mask_train_augmentation_path, "test_mask_" + "%s" %mask_counter + ".png")
                cv2.imwrite(maskRZ_path, maskRZ)

                bar.update(number)
    #return  augmented_image_tensor, augmented_mask_tensor