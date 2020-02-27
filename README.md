# MASK_RCNN_TCDD
In this project I am using the Mask R-CNN (https://github.com/matterport/Mask_RCNN) to perform Tree Crown Detection and Delineation (TCDD) in a tropical forest region.

Some tips

Install anaconda and create an enviroment, and in the enviroment

First, install all dependencies follwing the instructions in the web page https://github.com/matterport/Mask_RCNN.

Install the GDAL

Then, make the download of my packge.

I only made some code modifications to able the Mask R-CNN works with TIF images and shape files

You can use this to delineate tree crowns in other regions or to detect some tree specie.

Here you find the trained weights of my Mask R-CNN and an example present in the article.

Download the image with examples need to create the synthetic images
https://www.dropbox.com/s/0l6zg4exnqqbfnb/santagenebra_examples.tif?dl=0

Download the weights file to test the Mask R-CNN for tree crwon detetection and delineation
https://www.dropbox.com/s/bos9saqflmn9uz5/trees.h5?dl=0

Lines for change the path for your computer:

trees.py line 18.

forest.py between lines 847 - 857.

mergeshapefiles.py line 11.

To train:

python trees.py train --weights=path\to\weights --dataset=path\to\dataset --train_images=number_images_to_train --val_images=number_images_to_validation

To test:

python trees.py test --weights=path\to\weights --dataset=8 --test_images=9

After using the mergeshapefiles.py a file called response.shp will be created. You can see that the trees near of edge with other image can be splited in two parts. The to join the splited trees in one, you must use the polygonUnionUsingLines.R. This file is specific for a grid with 128*128 pixels, if in your computer you can run wiht bigger grid this code will not work. But you can get the idea from it R file and change for you.

In the polygonUnionUsingLines.R change the path in the lines 14, 15, 16 and 173 for the path in your computer
    
Go ahead and try.
