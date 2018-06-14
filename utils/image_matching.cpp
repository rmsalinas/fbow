#include <iostream>
#include <fstream>
#include <vector>

//
#include "vocabulary_creator.h"
// OpenCV
#include <opencv2/core/core.hpp>
using namespace std;

//command line parser
class CmdLineParser { int argc; char **argv; public: CmdLineParser(int _argc, char **_argv) :argc(_argc), argv(_argv) {}  bool operator[] (string param) { int idx = -1;  for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i;    return (idx != -1); } string operator()(string param, string defvalue = "-1") { int idx = -1;    for (int i = 0; i<argc && idx == -1; i++) if (string(argv[i]) == param) idx = i; if (idx == -1) return defvalue;   else  return (argv[idx + 1]); } };

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
void ImageMatching(vector<cv::Mat> &features, Vocabulary &voc, vector<map<double, int> > &scores, vector<string> &filenames, string outDir)
{

    fbow::fBow vv, vv2;
    int avgScore = 0;
    int counter = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i<features.size(); ++i)
    {
        vv = voc.transform(features[i]);
        map<double, int> score;
        for (int j = 0; j<features.size(); ++j)
        {

            vv2 = voc.transform(features[j]);
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
    for (int i = 0; i < scores.size(); i++)
    {
        std::stringstream str;

        command = string("cd ") + outDir + string("&");
        command += "mkdir ";
        str << i;
        command += str.str();
        command += "";
        system(command.c_str());

        command = "cd " + outDir + "&";
#ifdef WIN32
        command += "copy ";
#else
        command += "cp ";
#endif
        command += filenames[i];
        command += " ";
        command += str.str();
        command += "\\source.JPG";
        
        system(command.c_str());
        std::cout << command << std::endl;
        j = 0;
        for (auto it = scores[i].begin(); it != scores[i].end(); it++)
        {
            ++j;
            std::stringstream str2;
            command = string("cd ") + outDir + string("&");
#ifdef WIN32
            command += "copy ";
#else
            command += "cp ";
#endif
            command += filenames[it->second];
            command += " ";
            command += str.str();
            command += "\\";
            str2 << j << "-";
            str2 << it->first;
            command += str2.str();
            command += ".JPG";
            system(command.c_str());
            std::cout << command << std::endl;
        }

    }
    /*
    {
    cout<<vv.begin()->first<<" "<<vv.begin()->second<<endl;
    cout<<vv.rbegin()->first<<" "<<vv.rbegin()->second<<endl;
    }
    */
}

int main(int argc, char **argv)
{

    try {
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc<3) {
            cerr << "Usage:  features output.fbow [-k k] [-l L] [-t nthreads]" << endl;
            return -1;
        }


        string desc_name;
        auto features = readFeaturesFromYMLFile(argv[1], desc_name);

        cout << "DescName=" << desc_name << endl;
        const int k = stoi(cml("-k", "10"));
        const int L = stoi(cml("-l", "6"));
        const int nThreads = stoi(cml("-t", "1"));
        srand(0);
        fbow::VocabularyCreator voc_creator;
        fbow::Vocabulary voc;
        cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        voc_creator.create(voc, features, desc_name, fbow::VocabularyCreator::Params(k, L, nThreads));
        auto t_end = std::chrono::high_resolution_clock::now();
        cout << "time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << " msecs" << endl;
        cout << "vocabulary size : " << voc.size() << endl;
        cerr << "Saving " << argv[2] << endl;
        voc.saveToFile(argv[2]);

        vector<string> filenames;
        string outDir = argv[3];
        for (int i = 4; i < argc; ++i)
        {
            filenames.push_back(argv[i]);
        }

        vector<map<double, int> > scores;
        ImageMatching(features, voc, scores, filenames, outDir);

    }
    catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

    return 0;
}
