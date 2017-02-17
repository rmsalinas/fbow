//Second step,creates the vocabulary from the set of features. It can be slow
#include <iostream>
#include <fstream>
#include <vector>

//
#include "vocabulary_creator.h"
// OpenCV
#include <opencv2/core/core.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
using namespace std;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
vector<cv::Mat> readFeaturesFromFile(string filename){
vector<cv::Mat> features;
    //test it is not created
    std::ifstream ifile(filename);
    if (!ifile.is_open()){cerr<<"could not open input file"<<endl;exit(0);}
    uint32_t size;
    ifile.read((char*)&size,sizeof(size));
    features.resize(size);
    for(size_t i=0;i<size;i++){

        uint32_t cols,rows,type;
        ifile.read( (char*)&cols,sizeof(cols));
        ifile.read( (char*)&rows,sizeof(rows));
        ifile.read( (char*)&type,sizeof(type));
        features[i].create(rows,cols,type);
        ifile.read( (char*)features[i].ptr<uchar>(0),features[i].total()*features[i].elemSize());
    }
    return features;
}

// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<3){
            cerr<<"Usage:  features output.yml "<<endl;
            return -1;
        }


        auto features=readFeaturesFromFile(argv[1]);

        //defining terms for bowkmeans trainer
        cv::TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
          int dictionarySize = 1000;
          int retries = 1;
          int flags = cv::KMEANS_PP_CENTERS;
          cv::BOWKMeansTrainer bowTrainer(583,tc,retries,flags);
          for(auto &f:features){
              cv::Mat c32f;
              f.convertTo(c32f,CV_32F);
              bowTrainer.add(c32f);
          }
          cout<<"Clusgering"<<endl;
          cv::Mat dictionary=bowTrainer.cluster( );
          cout<<"done"<<endl;
          //store the vocabulary
          cv::FileStorage fs(argv[2], cv::FileStorage::WRITE);
          fs << "vocabulary" << dictionary;
          fs.release();

           cv::Mat uDictionary;
          dictionary.convertTo(uDictionary, CV_8UC1);
          cv::BOWImgDescriptorExtractor bowDE( cv::DescriptorMatcher::create("BruteForce-Hamming"));
          bowDE.setVocabulary(uDictionary);


          auto t_start=std::chrono::high_resolution_clock::now();
          cv::Mat res;
           for(int i=0;i<1000;i++)
               bowDE.compute(features[0],res);
            auto t_end=std::chrono::high_resolution_clock::now();


            cout<<"time="<<double(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count())/1e6<<" ns"<<endl;



    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}

