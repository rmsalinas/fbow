from os import walk
import os
import subprocess
#in order to use this script:
#1. Prepare a database
#2. Compile the code
#3. Create a directory for the results
#4. Set the configurations below
#5. Run this script by the following command
#   python ImageMatching.py
#---------------Configs---------------------
DBDIR = "" #the dir where images reside e.g. C:/database
DBNAME = "" #e.g. database (in the line above)
CODEDIR = "" #e.g. H:/fbow-windows/build/bin/Release
DESCRIPTOR = "akaze" #orb, surf
RESULTSDIR = ""#e.g. "C:/FBoWResults"
OS = "Windows" # or "LinuxBase"

#----------------Path Setting---------------
Extension = ".exe" if OS == "Windows" else ""
ENDCMD = "&" if OS == "Windows" else ";"
OUTPUT = RESULTSDIR + "\\" + DESCRIPTOR + DBNAME
MKDIRCMD = "mkdir " + OUTPUT
MatchMatrixFile = OUTPUT + "\\" + "MatchingMatrix.txt"

#---------------Commands--------------------
FeaturesExtractCmd = CODEDIR + "\\" + "fbow_create_voc_step0" + Extension + " " + DESCRIPTOR + " " + OUTPUT + "\\features " + DBDIR
VocabCreateCmd = CODEDIR + "\\" + "fbow_create_voc_step1" + Extension + " " + OUTPUT + "\\features " + OUTPUT + "\\out.fbow"
#ImageMatchingCmd = CODEDIR + "\\" + "image_matching" + Extension + " " + OUTPUT + "\\features " + OUTPUT + "\\out.fbow " + OUTPUT + " " + DBDIR + " " + MatchMatrixFile
ImageMatchingCmd = CODEDIR + "\\" + "image_matching" + Extension + " " + DESCRIPTOR + " " + DBDIR + " " + MatchMatrixFile + " " + OUTPUT

#---------------Run Commands----------------
print("cd " + RESULTSDIR + " " + ENDCMD + " " + "mkdir " + DESCRIPTOR + DBNAME)
print("\n")
os.system("cd " + RESULTSDIR + " " + ENDCMD + " " + "mkdir " + DESCRIPTOR + DBNAME)
print("\n")

# print(FeaturesExtractCmd)
# print("\n")
# os.system(FeaturesExtractCmd)
# print("\n")

print(ImageMatchingCmd)
print("\n")
os.system(ImageMatchingCmd)
print("\n")