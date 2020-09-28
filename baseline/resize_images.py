import os
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed
from skimage import exposure
import numpy as np
from tqdm import tqdm

inputfolder="source_data/data/" #enter the name of the folder where test and train folders are saved

output_folder="source_data/" #enter the directory where all the processed data is to be stored
#inside it 3 directories will be created to store, equalized, resized images, and npy files
#equalized images are already resized see line 39 if you don't want them resized


outputfolder_processed=os.path.join(output_folder,"processed/") #name of the output folder for processed images
outputfolder_resized=os.path.join(output_folder,"resized/") #name of the output folder for resized images
outputfolder_npy=os.path.join(output_folder,"npy/") #name of the output folder for npy files

#lists of test/benign test/malignant etc..
inputfolders=[os.path.join(inputfolder,i,j) for i in ["test/","train/"] for j in ["benign/","malignant/"]]
outputfolders_p=[os.path.join(outputfolder_processed,i,j) for i in ["test/","train/"] for j in ["benign/","malignant/"]]
outputfolders_r=[os.path.join(outputfolder_resized,i,j) for i in ["test/","train/"] for j in ["benign/","malignant/"]]

# this is for if output folders don't already exist, then to create them
for folder in outputfolders_p+outputfolders_r+[outputfolder_npy]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def resize_image(path, outfolder_p,outfolder_r,res):
    base_name = os.path.basename(path)
    outpath_processed = os.path.join(outfolder_p, base_name)
    outpath_resized= os.path.join(outfolder_r, base_name)
    image = Image.open(path)
    image_resized = image.resize(res,resample=Image.BILINEAR)
    
    eq_image = exposure.equalize_adapthist(np.asarray(image,dtype='uint8'),clip_limit=0.03)
    image_processed = Image.fromarray((eq_image*255).astype('uint8'),'RGB')
    #comment next line to not resize equalized images
    image_processed = image_processed.resize(res,resample=Image.BILINEAR)
    
    image_processed.save(outpath_processed)
    image_resized.save(outpath_resized)

def process_images(inputfolders,outputfolders_p,outputfolders_r,res):
    for i in range(4):
        images=glob.glob(os.path.join(inputfolders[i], "*.jpg"))
   
        Parallel(n_jobs=8)(
            delayed(resize_image)(
                img,outputfolders_p[i],outputfolders_r[i],res
            ) for img in tqdm(images, desc=" ".join(inputfolders[i].split('/')[-3:-1])+" images")
        )

def savenpyfiles(inputfolders,outfolder):
    for folder in inputfolders:
        #name for npy file is created based on where the data comes from
        npyfilename="_".join(folder.split('/')[-4:-1])+".npy"
        
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
        image_list = [read(os.path.join(folder, filename)) for filename in tqdm(os.listdir(folder),desc=npyfilename)]
        images = np.array(image_list, dtype='uint8')
        
        np.save(os.path.join(outfolder,npyfilename),images)

if __name__=="__main__":
    #comment and uncomment functions based on requirement
    process_images(inputfolders,outputfolders_p,outputfolders_r,res=(112,112)) 
    print("")
    savenpyfiles(outputfolders_p,outputfolder_npy) #save npy for processed files
    savenpyfiles(outputfolders_r,outputfolder_npy) #save npy for resized files
    savenpyfiles(inputfolders,outputfolder_npy)  #save npy for unprocessed files

