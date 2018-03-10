## set current Kaggle dir
setwd('/Users/mash/kdata')

dataFolderLoc = getwd()
folderImageID = dir(dataFolderLoc, "png",recursive=T)

write(folderImageID, "stuff.csv")

