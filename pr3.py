'''
Miguel Zavala
5/5/20
CISC442-Computer Vision
PR3-2

'''

'''
In this part we are working on the FCN network.
Do the following items:
1.Download 5  (or  more)  pictures  that  each  one  includes   at  least  3  different  
objects  from  these  classes: Aeroplan, Bicycle, Bird, Boat, Bottle, Bus, Car, Cat, Chair, 
Cow, Dining table, Dog, Horse, Motorbike, Person, Potted plant, Sheep, Sofa, Train, Tv/monitor
2.Similar to part 1, setup the Pytorch environment and create the pretrained FCN network using 
the fcn_resnet50 function.
3.For each image: •Feed the image to the network.•The network outputs 21 feature maps. Save all
 the feature maps in 1 image (as tiles).•Create the final segmentation image such that each color represents 1 class.•You might need to try different input sizes to get the best segmentation.   4.Write a paragraph about the feature maps.5.Submit your code along with a PDF file containing images, feature maps, final segmentation, and a paragraph writeup.
'''

#Part 1:import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from torchvision import transforms

print("main")

#PATHS---------------
IMAGEDIR = "./INPUT_IMAGES/"

#Takes a path directory (string) and checks for all images in that directory
#Returns a list of image paths (list of strings)
def getAllImagesFromInputImagesDir(path:str, getabspaths=True):
    listOfImagePaths = []
    if (path[0] == '.' and getabspaths):
        path = os.getcwd() + path[1:path.__len__()]

    # read the entries
    with os.scandir(path) as listOfEntries:
        curr_path = ""
        for entry in listOfEntries:
            # print all entries that are files
            if entry.is_file() and ('png' in entry.name.lower() or 'jpg' in entry.name.lower()):
                #print(entry.name)

                if (getabspaths):
                    curr_path=path+entry.name
                    #print(path)
                else:
                    curr_path = entry.name


                listOfImagePaths.append(curr_path)

    return listOfImagePaths

model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
#model=torchvision.models.segmentation.fcn_resnet50(pretrained=True)
#models.resnet50(pretrained=True)

model.eval()
#print(model.eval())

#Gets a list of images from our image directory:
listOfImages=getAllImagesFromInputImagesDir(IMAGEDIR,True)
print(listOfImages)

#Loops through our input images directory and displays the predicted classes and their feature maps:
for filename in listOfImages:
  print("Running current image:"+filename)

  #Loading our image:
  input_image = Image.open(filename)
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model



  #Checks if GPU is available:
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  #Gets the output of our res model:
  with torch.no_grad():
      output = model(input_batch)['out'][0] #Our 21 feature maps
      #print(len(output)) #Has a length of 21 for our 21 feature maps

  #Gets the best output image:
  final_predicted_classes_image = output.argmax(0)



  #Creates a unique color for each class:
  palette=torch.tensor([2**25-1,2**15-1,2**len(output)-1])
  colors=torch.as_tensor([i for i in range(len(output))])[:,None]*palette
  colors=(colors%255).numpy().astype("uint8")

  #Converts our Tensor final image into an np array:
  npimagearray_finalimage = Image.fromarray(final_predicted_classes_image.byte().cpu().numpy()).resize(input_image.size)

  #Applies a color to each found class within our final image:
  npimagearray_finalimage.putpalette(colors)

  #Displaying our final image:
  plt.axis('off')
  plt.imshow(npimagearray_finalimage)

  #Showcases our 21 feature maps in a single grid image:------
  _, axs = plt.subplots(5, 5, figsize=(6, 6))
  plt.axis('off')
  axs = axs.flatten()
  #Turns off the axis' for every tile:
  for i in axs:
    i.axis('off')
  i=0
  for img, ax in zip(output, axs):
      output_predictions = output[i]
      img = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
      #img.putpalette(colors)
      ax.imshow(img,cmap='gray',aspect='auto')
      i+=1
  plt.subplots_adjust(wspace=.01, hspace=.01)
  plt.show(block=True)
  #-----------------------------------------------------------  #-----------------------------------------------------------