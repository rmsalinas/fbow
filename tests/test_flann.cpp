
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include <cstdlib>
#include <memory>
using namespace std;
using namespace std;


std::vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")   fdetector=cv::ORB::create(2000);

    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2 );
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

// Include Opencv
#include <opencv2/flann.hpp>
#include "fbow.h"
#include <iostream>

// Namespaces
using namespace cv;
using namespace std;

namespace fbow{
class VocabularyCreator{
public:

static    cv::Mat getVocabularyLeafNodes(fbow::Vocabulary &voc){

    //analyze all blocks and count  the leafs
    uint32_t nleafs=0;
    for(uint32_t b=0;b<voc._params._nblocks;b++){
        fbow::Vocabulary::Block block=voc.getBlock(b);
        int nnodes=block.getN();
        for(int n=0;n<nnodes;n++)
            if (block.getBlockNodeInfo(n)->isleaf()) nleafs++;
    }
    //reserve memory
    cv::Mat features;
    if ( voc.getDescType()==CV_8UC1)
        features.create(nleafs,voc.getDescSize(),CV_8UC1);
    else
        features.create(nleafs,voc.getDescSize()/sizeof(float),CV_32FC1);
    //start copy data
    nleafs=0;
    for(uint32_t b=0;b<voc._params._nblocks;b++){
        fbow::Vocabulary::Block block=voc.getBlock(b);
        int nnodes=block.getN();
        for(int n=0;n<nnodes;n++)
            if (block.getBlockNodeInfo(n)->isleaf())  block.getFeature(n,features.row(nleafs++));
    }
    return features;
}
};
}


class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

// Main
int main(int argc, char** argv)
{
CmdLineParser cml(argc,argv);
    if(argc<3 || cml["-h"]) throw std::runtime_error ("Usage: fbow   image [descriptor]");

    fbow::Vocabulary voc;
    voc.readFromFile(argv[1]);
    string desc_name=voc.getDescName();
    cout<<"voc desc name="<<desc_name<<endl;
    if (argc>=4) desc_name=argv[3];

    auto features=loadFeatures({argv[2]},desc_name);
    cout<<"size="<<features[0].rows<<" "<<features[0].cols<<endl;
    auto voc_words=fbow::VocabularyCreator::getVocabularyLeafNodes(voc);
    cout<<"number of words="<<voc_words.rows<<endl;
    if(voc_words.type()!=voc.getDescType()){
        cerr<<"Invalid types for features according to the voc"<<endl;
        return -1;
    }
    std::shared_ptr<cv::flann::Index > tree;
    cout<<"Creating tree"<<endl;
    if (voc.getDescType()==CV_8UC1){
        cv::flann::HierarchicalClusteringIndexParams indexParams(voc.getK(),cvflann::FLANN_CENTERS_RANDOM,5,1) ;
        tree= std::make_shared<cv::flann::Index>(voc_words, indexParams,cvflann::FLANN_DIST_HAMMING);
    }
    else{
//        cv::flann::KDTreeIndexParams indexParams;
        cv::flann::KMeansIndexParams indexParams(voc.getK());
        cout<<voc.getK()<<endl;
        tree= std::make_shared<cv::flann::Index>(voc_words, indexParams);
    }

    cout<<"done"<<endl;

    cout << "Performing search to find nn to image features" << endl;
    cv::Mat indices;
    cv::Mat dists;
    auto t_start=std::chrono::high_resolution_clock::now();
    for(int i=0;i<1000;i++)
        tree->knnSearch(features[0], indices, dists, 1, cv::flann::SearchParams(1));
    auto t_end=std::chrono::high_resolution_clock::now();
    cout<<"time="<<double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count())<<" ms"<<endl;


}
