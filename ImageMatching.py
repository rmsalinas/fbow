from os import walk
import os
DBDIR = "Database dir" #the dir where images reside e.g. C:/database
DBNAME = "Databasename" #e.g. database (in the line above)
CODEDIR = "" #e.g. H:/fbow-windows/build/bin/Release
DESCRIPTOR = "orb"
RESULTSDIR = ""#e.g. "C:/FBoWResults"
OUTPUT = RESULTSDIR + "/" + DESCRIPTOR + DBNAME
MKDIRCMD = "mkdir " + OUTPUT
FeaturesExtractCmd = CODEDIR + "/" + "fbow_create_voc_step0.exe " + DESCRIPTOR + " " + OUTPUT + "/features"
VocabCreateCmd = CODEDIR + "/" + "fbow_create_voc_step1.exe " + OUTPUT + "/features " + OUTPUT + "/out.fbow"
ImageMatchingCmd = CODEDIR + "/" + "image_matching.exe " + OUTPUT + "/out.fbow " + OUTPUT + " "

imagesFileName = ""
i = 1
for (dirpath, dirnames, filenames) in walk(DBDIR):
    for filename in filenames:
        imagesFileName += " " + DBDIR + "/" + filename
    break
FeaturesExtractCmd += imagesFileName
ImageMatchingCmd += imagesFileName
print("cd " + RESULTSDIR + " & " + "mkdir " + DESCRIPTOR + DBNAME)
os.system("cd " + RESULTSDIR + " & " + "mkdir " + DESCRIPTOR + DBNAME)
print(FeaturesExtractCmd)
os.system(FeaturesExtractCmd)
print(VocabCreateCmd)
os.system(VocabCreateCmd)
print(ImageMatchingCmd)
os.system(ImageMatchingCmd)