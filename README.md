FBOW
=====
FBOW (Fast Bag of Words) is an extremmely optimized version of the DBow2/DBow3 libraries. The library is highly optimized to speed up the Bag of Words creation using  AVX,SSE and MMX instructions. In loading a vocabulary, fbow is ~80x faster than DBOW2 (see tests directory and try). In transforming an image into a bag of words using on machines with AVX instructions, it is ~6.4x faster.

## 
## Main features:
	* Only depends on OpenCV 
	* Any type of descriptors allowed out of the box (binary and real)
	* Dictionary creation from a set of images. Bugs found in DBOW2/3 corrected.
	* Extremmely fast bow creation using specialized versions using AVX,SSE and MMX instructions both for binary and floating point descriptors.
	* Very fast load of vocabularies

## 
## The main differences with DBOW2/3 are:

	* Not yet implemented indexing of images. 

##
## Citing

If use this project please cite

@online{Fbow,
  author = {Rafael Muñoz-Salinas},
  title = {{FBox} Fast Bag of Words},
  year = 2017,
  url = {https://github.com/rmsalinas/fbow},
  urldate = {2017-02-17}
}

##
## Vocabularies

In directory vocabularies you have the ORBSLAM2 vocabulary (https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary) in fbow format.
##
## Test speed
 Go to test and run the program test_dbow2VSfbow

