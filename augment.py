# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
#matplotlib inline

root='/home/cvbl/nb/dataset/div2klrx4'
fnames = glob.glob(os.path.join(root, '*'))
print(len(fnames),root )
    
for f in fnames:
    # reading the image using its path
    image = io.imread(f)
    imagename=os.path.basename(f)
    print(imagename)
    #os.mkdir('augdiv2klrx4')
    #flipLR = np.fliplr(image).save('save_dir/save_results/AB.jpg')
    Image.fromarray(np.fliplr(image)).save('augdiv2klrx4/'+imagename+'lr.png')
    Image.fromarray(np.flipud(image)).save('augdiv2klrx4/'+imagename+'up.png')
    Image.fromarray(image).save('augdiv2klrx4/'+imagename)
    
    #im.save("your_file.jpeg")

    # shape of the image
    
    #print(image.shape)
    # displaying the image
    #io.imshow(image)
    #break