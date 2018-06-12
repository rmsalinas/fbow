from os import walk
import os

#---------------Configs---------------------
DBDIR = "" #the dir where images reside e.g. C:/database
DBNAME = "" #e.g. database (in the line above)
CODEDIR = "" #e.g. H:/fbow-windows/build/bin/Release
DESCRIPTOR = "akaze"
RESULTSDIR = ""#e.g. "C:/FBoWResults"
OS = "Windows" # or "LinuxBase"

#----------------Path Setting---------------
Extension = ".exe" if OS == "Windows" else ""
ENDCMD = "&" if OS == "Windows" else ";"
OUTPUT = RESULTSDIR + "/" + DESCRIPTOR + DBNAME
MKDIRCMD = "mkdir " + OUTPUT

#---------------Commands--------------------
FeaturesExtractCmd = CODEDIR + "/" + "fbow_create_voc_step0" + Extension + " " + DESCRIPTOR + " " + OUTPUT + "/features.yml"
VocabCreateCmd = CODEDIR + "/" + "fbow_create_voc_step1" + Extension + " " + OUTPUT + "/features.yml " + OUTPUT + "/out.fbow"
ImageMatchingCmd = CODEDIR + "/" + "image_matching" + Extension + " " + OUTPUT + "/out.fbow " + OUTPUT + " "

#---------------List Images-----------------
imagesFileName = ""
i = 1
for (dirpath, dirnames, filenames) in walk(DBDIR):
    filenames.sort()
    for filename in filenames:
        imagesFileName += " " + DBDIR + "/" + filename
    break
FeaturesExtractCmd += imagesFileName
ImageMatchingCmd += imagesFileName

#---------------Run Commands----------------
print("cd " + RESULTSDIR + " " + ENDCMD + " " + "mkdir " + DESCRIPTOR + DBNAME)
os.system("cd " + RESULTSDIR + " " + ENDCMD + " " + "mkdir " + DESCRIPTOR + DBNAME)
print(FeaturesExtractCmd)
os.system(FeaturesExtractCmd)
print(VocabCreateCmd)
os.system(VocabCreateCmd)
print(ImageMatchingCmd)
os.system(ImageMatchingCmd)