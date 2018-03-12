Ther are two functions inside the python script "image_modules_512.py".
The first is called pre_processing() which combines the masks.
The second is called image_resizing() which will resize all the images and the masks to 512x512.

To use the functions do the following...
attach the folder "kaggle_image_manipulation" to the path or working directory then type the following:

import image_manipulation_512 as im

im.pre_processing(r"<Kaggle folder path>")
im.image_resizing(r"<Kaggle folder path>")


*The Kaggle folder path is just the folder with all of the data...
stage1_train
stage1_test
stage1_train_labels.csv
stage1_sample_submission.csv