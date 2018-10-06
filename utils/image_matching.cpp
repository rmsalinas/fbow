//loads a vocabulary, and a image. Extracts image feaures and then  compute the bow of the image
#include "fbow.h"
#include <iostream>
#include <map>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif


#include <chrono>
class CmdLineParser { int argc; char **argv; public: CmdLineParser(int _argc, char **_argv) :argc(_argc), argv(_argv) {}  bool operator[] (string param) { int idx = -1;  for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i;    return (idx != -1); } string operator()(string param, string defvalue = "-1") { int idx = -1;    for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i; if (idx == -1) return defvalue;   else  return (argv[idx + 1]); } };

vector< cv::Mat  >  loadFeatures(std::vector<string> path_to_images, string descriptor = "")  {
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")        fdetector = cv::ORB::create(2000);
    else if (descriptor == "brisk") fdetector = cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor == "akaze") fdetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if (descriptor == "surf")  fdetector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty())throw std::runtime_error("Could not open image" + path_to_images[i]);
        cout << "extracting features" << endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
    }
    return features;
}

int main(int argc, char **argv) {
    CmdLineParser cml(argc, argv);
    try {
        if (argc<3 || cml["-h"]) throw std::runtime_error("Usage: fbow   image [descriptor]");
        fbow::Vocabulary voc;
        voc.readFromFile(argv[1]);

        string desc_name = voc.getDescName();
        cout << "voc desc name=" << desc_name << endl;
        vector<vector<cv::Mat> > features(argc - 3);
        vector<map<double, int> > scores;
        vector<string > filenames(argc - 3);
        string outDir = argv[2];
        for (int i = 3; i < argc; ++i)
        {
            filenames[i - 3] = { argv[i] };
        }
        for (size_t i = 0; i<filenames.size(); ++i)
            features[i] = loadFeatures({ filenames[i] }, desc_name);

        fbow::fBow vv, vv2;
        int avgScore = 0;
        int counter = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i<features.size(); ++i)
        {
            vv = voc.transform(features[i][0]);
            map<double, int> score;
            for (size_t j = 0; j<features.size(); ++j)
            {

                vv2 = voc.transform(features[j][0]);
                double score1 = vv.score(vv, vv2);
                counter++;
                //		if(score1 > 0.01f)
                {
                    score.insert(pair<double, int>(score1, j));
                }
                printf("%f, ", score1);
            }
            printf("\n");
            scores.push_back(score);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        avgScore += double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());

        std::string command;
        int j = 0;
        for (size_t i = 0; i < scores.size(); i++)
        {
            std::stringstream str;

            command = "mkdir ";
            str << i;
            command += str.str();
            command += "/";
            system(command.c_str());

            command = "cp ";
            command += filenames[i];
            command += " ";
            command += str.str();
            command += "/source.JPG";
            
        system((string("cd ") + outDir).c_str());
            system(command.c_str());
            j = 0;
            for (auto it = scores[i].begin(); it != scores[i].end(); it++)
            {
                ++j;
                std::stringstream str2;
                command = "cp ";
                command += filenames[it->second];
                command += " ";
                command += str.str();
                command += "/";
                str2 << j << "-";
                str2 << it->first;
                command += str2.str();
                command += ".JPG";
                system(command.c_str());
            }

        }
        /*
        {
        cout<<vv.begin()->first<<" "<<vv.begin()->second<<endl;
        cout<<vv.rbegin()->first<<" "<<vv.rbegin()->second<<endl;
        }
        */
        std::cout << "avg score: " << avgScore << " # of features: " << features.size() << std::endl;
    }
    catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

}
