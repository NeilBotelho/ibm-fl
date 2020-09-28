# BASELINE MODEL
### Preprocessing
- Create a folder named source_data and add the data folder with train and test files. You can change the inputfolder variable in the script. 
- I'd suggest not changing the output_folder while using the script in this repository, since it's already in .gitignore. The variables are mainly for use in kaggle/colab.
- If you want to change the outdput directory, add all the data folders to .gitignore.
- The npy files will be availaible in source_data/npy, I've added the same path to jupyter notebook. You can change it according to your convinience.
- Add any other notebooks used for model training to .gitignore. Jupyter notebooks aren't friendly with git. 
- Do not edit the preprocessing notebook directly, make a copy and add it to .gitignore.
- Except notebooks and data, you can add any other scripts or files needed to be shared.
- The script requires joblib and tqdm. Install it before running the script.

<---Add any more documentation below this--->
