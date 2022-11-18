### My_projects
scripts for my projects

CV:
### remove duplicate images in a folder
duplicate_removal.py --dataset FOLDER --remove 1

### check if the images in the folders in the SUBSETS folder are also in the MAINFOLDER. In case there are some duplicates remove from the folders in the SUBSETS folder and can copy the rest such that all images in the MAINFOLDER will be unique. 
needle_dups.py --haystack MAINFOLDER --needles SUBSETS
