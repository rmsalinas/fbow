#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"

#include "fbow.h"
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
using ORBVocabulary=DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;

class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

std::vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") {
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


int main(int argc,char**argv){
    try{
        CmdLineParser cml(argc,argv);
        if(argc<4 || cml["-h"]) throw std::runtime_error ("Usage: dbowfile.txt   image  fbowfile.fbow ");
        cout<<"extracting features"<<endl;
        std::vector< cv::Mat  > features=loadFeatures({argv[2]}, "orb");




        double dbow2_load,dbow2_transform;
        double fbow_load,fbow_transform;

        {
            ORBVocabulary voc;
            cout<<"loading dbow2 voc...."<<endl;
            auto t_start=std::chrono::high_resolution_clock::now();
            voc.loadFromTextFile(argv[1]);
            auto t_end=std::chrono::high_resolution_clock::now();
            auto desc_vector=toDescriptorVector(features[0]);//transform into the mode required by dbow2
            dbow2_load=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            cout<<"load time="<<dbow2_load<<" ms"<<endl;
            cout<<"processing image 1000 times"<<endl;
            DBoW2::BowVector vv;
            t_start=std::chrono::high_resolution_clock::now();
            for(int i=0;i<1000;i++)
                voc.transform(desc_vector,vv);
            t_end=std::chrono::high_resolution_clock::now();

            cout<<vv.begin()->first<< " "<<vv.begin()->second<<endl;
            cout<<vv.rbegin()->first<< " "<<vv.rbegin()->second<<endl;
            dbow2_transform=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            cout<<"DBOW2 time="<<dbow2_transform<<" ms"<<endl;

        }
        //repeat with fbow

        fbow::Vocabulary fvoc;
        cout<<"loading fbow voc...."<<endl;
        auto t_start=std::chrono::high_resolution_clock::now();
        fvoc.readFromFile(argv[3]);
        auto t_end=std::chrono::high_resolution_clock::now();
        fbow_load=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
        cout<<"load time="<<fbow_load<<" ms"<<endl;

        {    cout<<"processing image 1000 times"<<endl;
            fbow::fBow vv;
            t_start=std::chrono::high_resolution_clock::now();
            for(int i=0;i<1000;i++){
                vv=fvoc.transform(features[0]);
            }
            t_end=std::chrono::high_resolution_clock::now();

            cout<<vv.begin()->first<< " "<<vv.begin()->second<<endl;
            cout<<vv.rbegin()->first<< " "<<vv.rbegin()->second<<endl;
            fbow_transform=double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count());
            cout<<"FBOW time="<<fbow_transform<<" ms"<<endl;
        }

        cout<<"Fbow load speed up="<<dbow2_load/ fbow_load<<" transform Speed up="<<dbow2_transform/fbow_transform<<endl;
    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }
}
