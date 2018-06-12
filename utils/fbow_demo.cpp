#include <iostream>
#include <vector>

// fbow
#include "vocabulary_creator.h"
#include "fbow.h"
#include "Database.h"

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

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
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

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
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

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features)
{
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    //const WeightingType weight = TF_IDF;
    //const ScoringType score = L1_NORM;
	const int nThreads = 1;

	srand(0);
	fbow::VocabularyCreator voc_creator;
	fbow::Vocabulary voc;
	cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
	voc_creator.create(voc, features, "orb", fbow::VocabularyCreator::Params(k, L, nThreads));

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
	fBow v1, v2;
    for(size_t i = 0; i < features.size(); i++)
    {
        v1 = voc.transform(features[i]);
        for(size_t j = 0; j < features.size(); j++)
        {
			v2 = voc.transform(features[j]);

			double score = fBow::score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.saveToFile("small_voc.yml.gz");
    cout << "Done" << endl;

	return;
}

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
	fbow::Vocabulary voc;
	voc.readFromFile("small_voc.yml.gz");

	Database db(voc, false, 0); // false = do not use direct index
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "database created!" << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);
        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
	cout << "database saved!" << endl;
}

void testLoadedDatabase(const  vector<cv::Mat > &features){

	// once saved, we can load it again
	cout << "Retrieving database once again..." << endl;
	Database db2("small_db.yml.gz");
	cout << "... loaded!  endl" << std::endl;
	//cout << "... done! This is: " << endl << db2 << endl;

	QueryResults ret;
	for (size_t i = 0; i < features.size(); i++)
	{
		db2.query(features[i], ret, 1);

		// ret[0] is always the same image in this case, because we added it to the
		// database. ret[1] is the second best match.

		QueryResults::iterator qit = ret.begin();
		cout << "Matching Image " << i << "-> id=" << qit->Id << " score=" << qit->Score << endl;
	}


}



// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor=argv[1];

        auto images=readImagePaths(argc,argv,2);
        vector< cv::Mat   >   features= loadFeatures(images,descriptor);
        testVocCreation(features);

        testDatabase(features);

		testLoadedDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
