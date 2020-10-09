import os
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

input_folder="source_data/isic_data/Images" #enter the name of the folder of images from isic archive

output_folder="source_data/isic_data" #enter the directory where the processed data is to be stored
output_folder=os.path.join(output_folder,"processed_images/") 

# this is for if output folders don't already exist, then to create them
if not os.path.exists(output_folder):
   os.makedirs(output_folder)

def resize_image(path,output_folder,res):
    #padding the images and changing the aspect ratio to 1:1
    def square_padded(image):
        width, height = image.size
        if width == height:
            return image
        elif width > height:
            result = Image.new(image.mode, (width, width), (0,0,0))
            result.paste(image, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(image.mode, (height, height), (0,0,0))
            result.paste(image, ((height - width) // 2, 0))
            return result
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    base_name = os.path.basename(path)
    outpath_processed = os.path.join(output_folder, base_name)
    image = Image.open(path)
    image = square_padded(image)
    image = image.resize(res)
    image.save(outpath_processed)

def process_images(input_folder,output_folder,res):
    images=glob.glob(os.path.join(input_folder, "*.jpeg"))
    Parallel(n_jobs=8)(
        delayed(resize_image)(
            img,output_folder,res
        ) for img in tqdm(images)
    )
#incase you wanted to save the images to npy, uncomment the function call in main
def savenpyfiles(input_folder,out_folder):
    npyfilename="isic_112.npy"
    
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    image_list = [read(os.path.join(input_folder, filename)) for filename in tqdm(os.listdir(input_folder),desc=npyfilename)]
    images = np.array(image_list, dtype='uint8')
    
    np.save(os.path.join(out_folder,npyfilename),images)

if __name__=="__main__":
    process_images(input_folder,output_folder,res=(112,112)) 
    outputfolder_npy="source_data/npy"
    #savenpyfiles(output_folder,outputfolder_npy)
