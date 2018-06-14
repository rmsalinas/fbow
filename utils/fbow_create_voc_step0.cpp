
//First step of creating a vocabulary is extracting features from a set of images. We save them to a file for next step
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include "fbow.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

using namespace fbow;
using namespace std;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( vector<string> path_to_images,string descriptor="") throw (exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create(2000);
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,  0,  3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if(image.empty())throw runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        cout << path_to_images[i] << " : " << descriptors.rows << endl;
        features.push_back(descriptors);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------
void saveToFile(string filename, const vector<cv::Mat> &features, string  desc_name, bool rewrite = true)throw (exception) {

    //test it is not created
    if (!rewrite) {
        fstream ifile(filename);
        if (ifile.is_open())//read size and rewrite
            runtime_error("ERROR::: Output File " + filename + " already exists!!!!!");
    }
    ofstream ofile(filename);
    if (!ofile.is_open()) { cerr << "could not open output file" << endl; exit(0); }

    char _desc_name[20];
    desc_name.resize(min(size_t(19), desc_name.size()));
    strcpy(_desc_name, desc_name.c_str());
    ofile.write(_desc_name, 20);

    uint32_t size = features.size();
    ofile.write((char*)&size, sizeof(size));
    int i = 0;
    for (auto &f : features) {
        if (!f.isContinuous()) {
            cerr << "Matrices should be continuous" << endl; exit(0);
        }
        uint32_t aux = f.cols; ofile.write((char*)&aux, sizeof(aux));
        cout << "i : " << i++ << "rows : " << f.rows << endl;
        aux = f.rows; ofile.write((char*)&aux, sizeof(aux));
        aux = f.type(); ofile.write((char*)&aux, sizeof(aux));
        ofile.write((char*)f.ptr<uchar>(0), f.total()*f.elemSize());
    }
}

// ----------------------------------------------------------------------------
void saveToYMLFile(string filename, const vector<cv::Mat> &features, string  desc_name, bool rewrite = true)throw (exception) {

    //test it is not created
    if (!rewrite) {
        fstream ifile(filename);
        if (ifile.is_open())//read size and rewrite
            runtime_error("ERROR::: Output File " + filename + " already exists!!!!!");
    }

    cv::FileStorage file(filename, cv::FileStorage::WRITE);
    if (!file.isOpened()) { cerr << "could not open output file" << endl; exit(0); }

    // Declare what you need
    file << "descriptor name" << desc_name;
    file << "num features" << (int)(features.size());//does not handle size_t for some reason!
    for (int i = 0; i < features.size();++i) {
        if (!features[i].isContinuous()) {
           cerr << "Matrices should be continuous" << endl; exit(0);
        }
        stringstream str;
        str << "featureidx" << i;
        // Write to file!
        file << str.str() << features[i];
    }
    file.release();
}

// ----------------------------------------------------------------------------
vector<string> ListFilenames(string dbPath)
{
    namespace fs = std::experimental::filesystem;
    vector<string> filenames;
    for (auto & p : fs::directory_iterator(dbPath))
    {
        stringstream str;
        str << p;
        filenames.push_back(str.str());

    }
    return filenames;
}

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc==1){
            cerr<<"Usage:  descriptor_name output image0 image1 ... \n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)"<<endl;
            return -1;
        }

        string descriptor=argv[1];
        string output=argv[2];

        auto images = ListFilenames(argv[3]);
        //auto images=readImagePaths(argc,argv,3);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);

        //save features to file
        saveToYMLFile(argv[2],features,descriptor);

    }catch(exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
