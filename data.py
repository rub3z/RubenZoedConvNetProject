'''
    This file is used to create numpy happy data out of all of the images used.
    The alexNetWanabe.py used this file for convinience and to be able to just 
    load the dataset instead of using a real time solution that might prolong the 
    already lengthy excecution of training a model.
'''


import numpy as np
from os import listdir
from matplotlib import pyplot as plot

''' Iterrates through out 100 x 100 x 3 images translates them into matrix form'''
def loadImages(path):
    imagesList = listdir(path)
    loadedImages = np.zeros([0,100,100,3])

    #traverse choosen directory
    for image in imagesList:
        if image.find(".jp") > 0:
            img = plot.imread(path + image)#PImage.open(path + image)

            #incase we have black and white images or images with an alpha value, reshape to proper form
            if img.shape  == (100, 100) or img.shape == (100,100,4):
                img = np.resize(img, (100,100,3))
            
            #turn it into a numpy array
            arr = np.array([np.array(img)])
                
            #added to the list    
            loadedImages = np.append(loadedImages, arr, axis = 0)
        
    return loadedImages


'''Gathers the images/data we need for our machine learning process and returns it in two list.  
   combinedImageSetholds the matrix data of the images
   combinedLabelSet holds the numpy array of all the labels pertaining to the images'''
def getData(mainPath)
    
    drawings = loadImages(mainPath + "drawings/")
    count = drawings.shape[0]
    drawingLabels = np.full((count,1), 0, int)   #creating the labels, drawings will be known ass class 0
    print("Done with Drawings")
    
    engravings = loadImages(mainPath + "engraving/")
    count = engravings.shape[0]
    engravingLabels = np.full((count,1), 1, int)   #creating the labels, engravings will be known ass class 1
    print("Done with Engravings")
    
    iconographys = loadImages(mainPath + "iconography/")
    count = iconographys.shape[0]
    iconographyLabels = np.full((count,1), 2, int)   #creating the labels, iconographies will be known ass class 2
    print("Done with Iconography")
    
    paintings = loadImages(mainPath + "painting/")
    count = paintings.shape[0]
    paintingLabels = np.full((count,1), 3, int)   #creating the labels, paintings will be known ass class 3
    print("Done with Painting")
    
    sculptures = loadImages(mainPath + "sculpture/")
    count = sculptures.shape[0]
    sculptureLabels = np.full((count,1), 4, int)   #creating the labels, sculptures will be known ass class 4
    print("Done with Sculpture")
    
    #Combining all images and all labels into two different arrays that will be saved in two different files
    combinedImageSet = drawings
    combinedLabelSet = drawingLabels
    
    combinedImageSet = np.append(combinedImageSet, engravings, axis = 0)
    combinedLabelSet = np.append(combinedLabelSet, engravingLabels, axis = 0)
    
    combinedImageSet = np.append(combinedImageSet, iconographys, axis = 0)
    combinedLabelSet = np.append(combinedLabelSet, iconographyLabels, axis = 0)
    
    combinedImageSet = np.append(combinedImageSet, paintings, axis = 0)
    combinedLabelSet = np.append(combinedLabelSet, paintingLabels, axis = 0)
    
    combinedImageSet = np.append(combinedImageSet, sculptures, axis = 0)
    combinedLabelSet = np.append(combinedLabelSet, sculptureLabels, axis = 0)

    return combinedImageSet, combinedLabelSet
    

    
'''Saves all of the training images in matrix format and also the training labels'''
def getTrainingData():
    images, labels = getData('''location of training data''')
    np.save("ImageSet-100", images)
    np.save("LabelSet-100", labels)



'''Saves all of the validation images in matrix format and also the validation labels'''
def getValidationData():
    images, labels = getData('''location of validation data''')
    np.save("ImageSet-100", images)
    np.save("LabelSet-100", labels)


    
### Run the code to save the data ###
getTrainingData()
getValidationData()





