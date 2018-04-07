def pre_processing(kaggle_folder_path):
    import numpy as np
    import pathlib
    import cv2
    import os

    #This section will define all the starting paths that will be needed
    stage1_train_path = os.path.join(kaggle_folder_path, "stage1_train")
    stage1_test_path = os.path.join(kaggle_folder_path, "stage1_test")
    #This builds paths to the augmented train data folders we want to create
    training_data_augmentation_path = os.path.join(kaggle_folder_path, "training_data_augmentation")
    image_train_augmentation_path = os.path.join(training_data_augmentation_path, "image_train_augmentation")
    mask_train_augmentation_path = os.path.join(training_data_augmentation_path, "mask_train_augmentation")
    #This builds a path to a new folder where we will keep all testing images in one place
    testing_images_path = os.path.join(kaggle_folder_path, "testing_images")
    #This part does the actual building of the folders
    if not os.path.exists(training_data_augmentation_path):
        os.makedirs(training_data_augmentation_path)
    if not os.path.exists(image_train_augmentation_path):
        os.makedirs(image_train_augmentation_path)
    if not os.path.exists(mask_train_augmentation_path):
        os.makedirs(mask_train_augmentation_path)
    if not os.path.exists(testing_images_path):
       os.makedirs(testing_images_path) 

    
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
                #print(stage1_train_images_list_path)
                move_image_train_path = os.path.join(image_and_masks_train_path, "image.png")
                cv2.imwrite(move_image_train_path, original_img_train)    
                                   
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
                #print(stage1_train_masks_list_path)
                #this will put all_masks_train image and save it into the file called combined_masks
                cv2.imwrite(combined_masks_path, all_masks_train)
    
    #Now start building the test folders up
    counter = 1
    for folder_test in os.listdir(stage1_test_path):
        #os.listdir(path) will list the items in the directory
        stage1_test_folder_path = "%s" % folder_test
        stage1_test_folder_path = os.path.join(stage1_test_path, stage1_test_folder_path,"images")

        for items_test in os.listdir(stage1_test_folder_path): 
            stage1_test_folder_images_path ="%s" % items_test
            stage1_test_folder_images_path = os.path.join(stage1_test_folder_path, stage1_test_folder_images_path)        
            original_img_test = cv2.imread(stage1_test_folder_images_path,1)
            move_image_test_path = "test_image" + "%s" % counter + ".png"
            move_image_test_path = os.path.join(testing_images_path, move_image_test_path )
            cv2.imwrite(move_image_test_path, original_img_test)    
        counter = counter + 1