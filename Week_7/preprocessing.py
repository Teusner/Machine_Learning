
#%% package
import numpy as np
import matplotlib.pyplot as plt

from loadImages import loadImages
from selectFeatureVectors import selectFeatureVectors
from displayFeatures2d import displayFeatures2d
from displayFeatures3d import displayFeatures3d


#%% def preprocessing
def preprocessing():
    
    img73, img87 = loadImages()
    featLearn = selectFeatureVectors(img73, 500)

    return featLearn, img73, img87
    