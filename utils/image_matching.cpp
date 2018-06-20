#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

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
    std::ifstream ifile(filename, ios_base::binary);
    if (!ifile.is_open()) { cerr << "could not open input file" << endl; exit(0); }
    

    char _desc_name[20];
    ifile.read(_desc_name, 20);
    desc_name = _desc_name;

    uint32_t size;
    ifile.read((char*)&size, sizeof(size));
    ifile.close();
    features.resize(size);
    vector<vector<char> > tmp;
    tmp.resize(size);
    for (size_t i = 0; i<size; i++) {
        stringstream str;
        str << i;
        string fname = filename + str.str();
        std::ifstream ifile(fname, ios_base::binary);
        uint32_t cols, rows, type;
        ifile.read((char*)&cols, sizeof(cols));
        ifile.read((char*)&rows, sizeof(rows));
        ifile.read((char*)&type, sizeof(type));
        features[i].release();
        features[i].create(rows, cols, type);
        ifile.read((char*)features[i].data, features[i].total() * features[i].elemSize());
        ifile.close();
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
void ImageMatching(vector<cv::Mat> &features, Vocabulary &voc, vector<map<double, int> > &scores, vector<string> &filenames, string outDir, string MatchMatrixFile)
{

    fbow::fBow vv, vv2;
    int avgScore = 0;
    int counter = 0;
    auto t_start = std::chrono::high_resolution_clock::now();
    ofstream fstr(MatchMatrixFile);
    if (fstr.is_open() == false) {std::cerr << " error openning file : " << MatchMatrixFile << endl;}
    stringstream sstr;
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
            fstr << score1 << "\t";
        }
        fstr << endl;
        scores.push_back(score);
    }
    fstr.close();
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
//-------------------------------------------------------------------------------
vector< cv::Mat  >  loadFeatures(vector<string> path_to_images, string descriptor = "") throw (exception) {
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

    else throw runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for (size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout << "reading image: " << path_to_images[i] << endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        if (image.empty())throw runtime_error("Could not open image" + path_to_images[i]);
        cout << "extracting features" << endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        cout << path_to_images[i] << " : " << descriptors.rows << endl;
        features.push_back(descriptors);
        cout << "done detecting features" << endl;
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
        ifile.close();
    }

    ofstream ofile(filename, ios_base::binary);
    if (!ofile.is_open()) { cerr << "could not open output file" << endl; exit(0); }

    char _desc_name[20];
    desc_name.resize(min(size_t(19), desc_name.size()));
    strcpy(_desc_name, desc_name.c_str());
    ofile.write(_desc_name, 20);

    uint32_t size = features.size();
    ofile.write((char*)&size, sizeof(size));
    ofile.close();
    
    int i = 0;
    for (auto &f : features) {
        stringstream str;
        str << i;
        ofstream ffile(filename + str.str(), ios_base::binary);
        if (!f.isContinuous()) {
            cerr << "Matrices should be continuous" << endl; exit(0);
        }
        uint32_t aux = f.cols; ffile.write((char*)&aux, sizeof(aux));
        aux = f.rows; ffile.write((char*)&aux, sizeof(aux));
        aux = f.type(); ffile.write((char*)&aux, sizeof(aux));
        ffile.write((char*)f.data, f.total()*f.elemSize());

        ffile.close();
        i++;
    }
}


int main(int argc, char **argv)
{

    try {
        CmdLineParser cml(argc, argv);
        if (cml["-h"] || argc<3) {
            cerr << "Usage:  features output.fbow [-k k] [-l L] [-t nthreads]" << endl;
            return -1;
        }


        //string desc_name;
        //string outDir = argv[3];
        //string DBDir = argv[4];
        //string MatchMatrixFile = argv[5];
        //vector<string> filenames;

        //filenames = ListFilenames(DBDir);

        ////auto features = readFeaturesFromYMLFile(argv[1], desc_name);
        //auto features = readFeaturesFromFile(argv[1], desc_name);
        //cout << "DescName=" << desc_name << endl;
        //const int k = stoi(cml("-k", "10"));
        //const int L = stoi(cml("-l", "6"));
        //const int nThreads = stoi(cml("-t", "1"));
        //srand(0);
        //fbow::VocabularyCreator voc_creator;
        //fbow::Vocabulary voc;
        //cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
        //auto t_start = std::chrono::high_resolution_clock::now();
        //voc_creator.create(voc, features, desc_name, fbow::VocabularyCreator::Params(k, L, nThreads));
        //auto t_end = std::chrono::high_resolution_clock::now();
        //cout << "time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) << " msecs" << endl;
        //cout << "vocabulary size : " << voc.size() << endl;
        //cerr << "Saving " << argv[2] << endl;
        //voc.saveToFile(argv[2]);

        //akaze DBadress Matrix output
        string descriptor;
        string MatchMatrixFile = argv[3];
        string outDir = argv[4];
        fbow::Vocabulary voc;
        fbow::VocabularyCreator voc_creator;
        const int k = stoi(cml("-k", "10"));
        const int L = stoi(cml("-l", "6"));
        const int nThreads = stoi(cml("-t", "1"));

        auto images = ListFilenames(argv[2]);
        //auto images=readImagePaths(argc,argv,3);
        //vector<cv::Mat>   features = loadFeatures(images, descriptor);
        //saveToFile(argv[2], features, descriptor);
        vector<cv::Mat>   features = readFeaturesFromFile(argv[1], descriptor);

        voc_creator.create(voc, features, descriptor, fbow::VocabularyCreator::Params(k, L, nThreads));
        vector<map<double, int> > scores;
        ImageMatching(features, voc, scores, images, outDir, MatchMatrixFile);
        

    }
    catch (std::exception &ex) {
        cerr << ex.what() << endl;
    }

    return 0;
}
