import numpy as np
import os
import pathlib
import cv2

def pre_processing(kaggle_folder_path):

    stage1_train = "stage1_train"
    stage1_train_path = os.path.join(kaggle_folder_path, stage1_train)

    max_size = [0,0]

    for folder in os.listdir(stage1_train_path):
        
        #os.listdir(path) will list the items in the directory
        f1_path = "%s" % folder
        f1_path = os.path.join(stage1_train_path, f1_path)
        #this step concatenates the path to a new path
        
        
        for items in os.listdir(f1_path):
            
            if items =="images":
                f2_0_path ="%s" % items
                f2_0_path = os.path.join(f1_path,f2_0_path)
                for images1 in os.listdir(f2_0_path):
                    f2_1_path = "%s" % images1
                    f2_1_path = os.path.join(f2_0_path,f2_1_path)
                    origional_img = cv2.imread(f2_1_path,0)
                    height,width = np.shape(origional_img)
            
            if items == "masks":
                all_masks= np.zeros([height,width,1],np.uint8)
                #this creates a blank image to start with for the adding of masks
                f2_path = "%s" % items
                f2_path = os.path.join(f1_path,f2_path)
                for images in os.listdir(f2_path):
                    f3_path = "%s" % images
                    f3_path = os.path.join(f2_path,f3_path)
                    picture = cv2.imread(f3_path,0)
                    all_masks = cv2.add(all_masks,picture)

                mask_name = "%s" % "masks_added.png"
                combined_masks = os.path.join(f1_path,mask_name)

                #this will put all_masks image and save it into the file called combined_masks
                cv2.imwrite(combined_masks,all_masks)
               
            if height > max_size[0]:
                max_size[0] = height
            if width > max_size[1]:
                max_size[1] = width

###########################start padding section#####################


def image_resizing(kaggle_folder_path):



    stage1_train = "stage1_train"
    folder_path = os.path.join(kaggle_folder_path, stage1_train)
    Padded_Folder = "Padded_Folder"
    padded_folder_path = os.path.join(kaggle_folder_path, Padded_Folder)


    for folder1 in os.listdir(folder_path):
        
        stage1_train_folder_path = "%s" % folder1
        stage1_train_folder_path = os.path.join(folder_path, stage1_train_folder_path)
        padded_stage1_train_folder_path = "%s" % folder1
        padded_stage1_train_folder_path = os.path.join(padded_folder_path, padded_stage1_train_folder_path)
        
        for folder2 in os.listdir(stage1_train_folder_path):
            if folder2 == "images":
                
                stage1_train_folder_images_path = "%s" % folder2
                stage1_train_folder_images_path = os.path.join(stage1_train_folder_path,stage1_train_folder_images_path)
                padded_stage1_train_folder_images_path = "%s" % folder2
                padded_stage1_train_folder_images_path = os.path.join(padded_stage1_train_folder_path, padded_stage1_train_folder_images_path)

                #now need to make a file to hold padded images
                pathlib.Path(padded_stage1_train_folder_images_path).mkdir(parents=True, exist_ok=True)
                
                for images in os.listdir(stage1_train_folder_images_path):
                    
                    stage1_train_folder_images_file_path = "%s" % images
                    stage1_train_folder_images_file_path = os.path.join(stage1_train_folder_images_path,stage1_train_folder_images_file_path)
                    padded_stage1_train_folder_images_file_path = "%s" % images
                    padded_stage1_train_folder_images_file_path = os.path.join(padded_stage1_train_folder_images_path,padded_stage1_train_folder_images_file_path)
                    
                    original_img = cv2.imread(stage1_train_folder_images_file_path,0)
                  
                    resized_image = cv2.resize(original_img, (512,512))
                     #now need to write this padded mask image to a folder
                    cv2.imwrite(padded_stage1_train_folder_images_file_path, resized_image) 
                    
            
            if folder2 == "masks":
                
                stage1_train_folder_masks_path ="%s" % folder2
                stage1_train_folder_masks_path = os.path.join(stage1_train_folder_path,stage1_train_folder_masks_path)
                padded_stage1_train_folder_masks_path ="%s" % folder2
                padded_stage1_train_folder_masks_path = os.path.join(padded_stage1_train_folder_path,padded_stage1_train_folder_masks_path)
                #now need to make a file to hold padded masks
                pathlib.Path(padded_stage1_train_folder_masks_path).mkdir(parents=True, exist_ok=True)

                for masks in os.listdir(stage1_train_folder_masks_path):
                    
                    stage1_train_folder_masks_file_path = "%s" % masks
                    stage1_train_folder_masks_file_path = os.path.join(stage1_train_folder_masks_path,stage1_train_folder_masks_file_path)
                    padded_stage1_train_folder_masks_file_path = "%s" % masks
                    padded_stage1_train_folder_masks_file_path = os.path.join(padded_stage1_train_folder_masks_path ,padded_stage1_train_folder_masks_file_path)

                    mask_img = cv2.imread(stage1_train_folder_masks_file_path,0) 
                    
                    resized_mask = cv2.resize(mask_img, (512,512))
                    #now need to write this padded mask image to a folder
                    cv2.imwrite(padded_stage1_train_folder_masks_file_path, resized_mask)
                