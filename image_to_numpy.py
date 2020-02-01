from PIL import Image
import numpy as np
import os

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )


path = os.getcwd() + '/processed_celeba_small/celeba'

files = os.listdir(path)

print(len(files))
