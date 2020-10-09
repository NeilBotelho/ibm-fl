import os
import glob
from PIL import Image, ImageFile
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

input_folder="source_data/isic_data/Descriptions" #enter the name of the folder where test and train folders are saved

output_file="source_data/data.pickle" #enter the directory where all the processed data is to be stored

def main():
    descriptions=glob.glob(os.path.join(input_folder, "ISIC*"))

    df=pd.DataFrame(data=None,columns=("name","age_approx","anatom_general_site","sex","image","image_name","labels"))

    for desc in tqdm(descriptions):
        desc=json.load(open(desc))
        image_name=desc["name"]+".jpeg"
        image_path="source_data/isic_data/processed_images/"+image_name
        data_dict={
            "name": desc["name"],
            "age_approx":desc["meta"]["clinical"]["age_approx"],
            "anatom_general_site": desc["meta"]["clinical"]["anatom_site_general"],
            "sex":desc["meta"]["clinical"]["sex"],
            "labels":desc["meta"]["clinical"]["benign_malignant"],
            "image": Image.open(image_path),
            "image_name":image_name
        }
        df=df.append(data_dict,ignore_index=True)
    return df

if __name__=="__main__":
    df=main()
    df.to_pickle(output_file)