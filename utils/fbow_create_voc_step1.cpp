//Second step,creates the vocabulary from the set of features. It can be slow
#include <iostream>
#include <fstream>
#include <vector>

//
#include "vocabulary_creator.h"
// OpenCV
#include <opencv2/core/core.hpp>
using namespace std;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
vector<cv::Mat> readFeaturesFromFile(string filename, std::string &desc_name) {
    vector<cv::Mat> features;
    //test it is not created
    std::ifstream ifile(filename);
    if (!ifile.is_open()) { cerr << "could not open input file" << endl; exit(0); }


    char _desc_name[20];
    ifile.read(_desc_name, 20);
    desc_name = _desc_name;

    uint32_t size;
    ifile.read((char*)&size, sizeof(size));
    features.resize(size);
    for (size_t i = 0; i<size; i++) {

        uint32_t cols, rows, type;
        ifile.read((char*)&cols, sizeof(cols));
        ifile.read((char*)&rows, sizeof(rows));
        std::cout << " i : " << i << " rows : " << rows << std::endl;
        ifile.read((char*)&type, sizeof(type));
        features[i].create(rows, cols, type);
        ifile.read((char*)features[i].ptr<uchar>(0), features[i].total()*features[i].elemSize());
    }
    return features;
}

vector<cv::Mat> readFeaturesFromYMLFile(string filename, std::string &desc_name) {
    vector<cv::Mat> features;
    cv::FileStorage file(filename, cv::FileStorage::READ);
    //test it is not created
    std::ifstream ifile(filename);
    if (!file.isOpened()) { cerr << "could not open input file" << endl; exit(0); }
    int size;

    file["descriptor name"] >> desc_name;
    file["num features"] >> size;//does not support uint32_t
    features.resize(size);
    for (size_t i = 0; i<size; i++) {
        stringstream str;
        str << "featureidx" << i;
        file[str.str()] >> features[i];
        
    }
    return features;
}

// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<3){
            cerr<<"Usage:  features output.fbow [-k k] [-l L] [-t nthreads]"<<endl;
            return -1;
        }


        string desc_name;
        auto features= readFeaturesFromYMLFile(argv[1],desc_name);

        cout<<"DescName="<<desc_name<<endl;
        const int k = stoi(cml("-k","10"));
        const int L = stoi(cml("-l","6"));
        const int nThreads=stoi(cml("-t","1"));
        srand(0);
        fbow::VocabularyCreator voc_creator;
        fbow::Vocabulary voc;
        cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
        auto t_start=std::chrono::high_resolution_clock::now();
        voc_creator.create(voc,features,desc_name, fbow::VocabularyCreator::Params(k, L,nThreads));
        auto t_end=std::chrono::high_resolution_clock::now();
        cout<<"time="<<double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count())<<" msecs"<<endl;
        cout<<"nblocks="<<voc.size()<<endl;
        cerr<<"Saving "<<argv[2]<<endl;
        voc.saveToFile(argv[2]);


    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
