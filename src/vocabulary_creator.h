#ifndef _FBOW_VOCABULARYCREATOR_H
#define _FBOW_VOCABULARYCREATOR_H
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>
#include <list>
#include <limits>
#include <cstdint>
#include <functional>
#include <opencv2/core/core.hpp>
#include "fbow_exports.h"
#include "fbow.h"
namespace fbow{
/**This class creates the vocabulary
 */
class FBOW_API VocabularyCreator
{
public:
    struct Params{
        Params(){}
        Params(uint32_t K ,int l=-1,uint32_t Nthreads=1,int MaxIters=-2):k(K),L(l),nthreads(Nthreads){
            if ( MaxIters!=-2) maxIters=MaxIters;
        }
        uint32_t k=32;
        int L=-1;
        uint32_t nthreads=1;
        int maxIters=11;
        bool verbose=false;
    };

    //create this from a set of features
    //Voc resulting vocabulary
    //features: vector of features. Each matrix represents the features of an image.
    //k braching factor
    //L maximum tree depth

    void create(fbow::Vocabulary &Voc, const std::vector<cv::Mat> &features, const std::string &desc_name, Params params);
    void create(fbow::Vocabulary &Voc, const cv::Mat &features, const std::string &desc_name, Params params);
private:
    Params _params;
    struct feature_info{
        feature_info(uint32_t MIDX,uint32_t FIDX):midx(MIDX),fidx(FIDX){}
        uint32_t midx,fidx;//matrix feature into the vector of matrices and fidx feature into the matrix
        float m_Dist;
        uint32_t parent=0;
    };
    //struct to acces the features as a unique vector
    struct FeatureInfo{
        void create(const std::vector<cv::Mat> &f)
        {
            features=(std::vector<cv::Mat>*)&f;
            uint32_t _size=0;
            for(auto &m:*features) _size+=m.rows;
            fInfo.clear();
            fInfo.reserve(_size);
            for(size_t midx=0;midx<features->size();midx++){
                auto nrows=features->at(midx).rows;
                for(int i=0;i<nrows;i++)
                    fInfo.push_back(feature_info(midx,i));
            }
        }
        size_t size()const {return fInfo.size();}
        template<typename T>
        inline T*getFeaturePtr(uint32_t i){const auto &idx=fInfo[i]; return features->at(idx.midx).ptr<T>(idx.fidx);}
        inline cv::Mat operator[](uint32_t i){const auto &idx=fInfo[i]; return features->at(idx.midx).row(idx.fidx);}
        inline feature_info & operator()(uint32_t i){ return fInfo[i];}


        std::vector<cv::Mat> *features;
        std::vector<feature_info> fInfo;
    };



    cv::Mat meanValue_binary( const std::vector<uint32_t>  &indices);
    cv::Mat meanValue_float( const std::vector<uint32_t>  &indices);

    void createLevel(const std::vector<uint32_t> &findices,  int parent=0, int curL=0);
    void createLevel(int parent=0, int curL=0, bool recursive=true);
    std::vector<uint32_t> getInitialClusterCenters(const std::vector<uint32_t> &findices);

    std::size_t vhash(const std::vector<std::vector<uint32_t> >& v_vec)  ;


    void thread_consumer(int idx);
    //thread sage queue to implement producer-consumer
    template <typename T>
    class Queue
    {
    public:

        T pop()
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            while (queue_.empty())
            {
                cond_.wait(mlock);
            }
            auto item = queue_.front();
            queue_.pop();
            return item;
        }

        void push(const T& item)
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            queue_.push(item);
            mlock.unlock();
            cond_.notify_one();
        }

        size_t size()
        {
            std::unique_lock<std::mutex> mlock(mutex_);
            size_t s=queue_.size();
            return s;
        }
    private:
        std::queue<T> queue_;
        std::mutex mutex_;
        std::condition_variable cond_;
    };

    Queue<std::pair<int,int> > ParentDepth_ProcesQueue;//queue of parent to be processed
    //used to retain the distance between each pair of nodes

    inline  uint64_t join(uint32_t a ,uint32_t b){
        uint64_t a_b;
        uint32_t *_a_b_16=(uint32_t*)&a_b;
        if( a>b) {
            _a_b_16[0]=b;_a_b_16[1]=a;
        }
        else{
            _a_b_16[1]=b;_a_b_16[0]=a;
        }
        return a_b;
    }
    inline std::pair<uint32_t,uint32_t> separe(uint64_t a_b){         uint32_t *_a_b_16=(uint32_t*)&a_b;return std::make_pair(_a_b_16[1],_a_b_16[0]);}

    //for each pair of nodes, their distance
    //   std::map<uint64_t,float> features_distance;

    //
    struct Node{
        Node(){}
        Node(uint32_t Id,uint32_t Parent,const cv::Mat &Feature, uint32_t Feat_idx=std::numeric_limits<uint32_t>::max() ):id(Id),parent(Parent),feature(Feature),feat_idx(Feat_idx){

        }

        uint32_t id=std::numeric_limits<uint32_t>::max();//id of this node in the tree
        uint32_t parent=std::numeric_limits<uint32_t>::max();//id of the parent node
        cv::Mat feature;//feature of this node
        //index of the feature this node represent(only if leaf and it stop because not enough points to create a new leave.
        //In case the node is a terminal point, but has many points beloging to its cluster, then, this is not set.
        //In other words, it is only used in nn search problems where L=-1

        uint32_t feat_idx=std::numeric_limits<uint32_t>::max();
        std::vector<uint32_t> children;
        bool isLeaf()const{return children.size()==0;}
        //if leaf, its weight and the word id
        float weight=1;
    };



    class Tree{
    public:
        Tree(){
            //add parent
            Node n;n.id=0;
            _nodes.insert(std::make_pair(0,n));
        }
        void add(const std::vector<Node> &new_nodes,int parentId){
            std::unique_lock<std::mutex> mlock(mutex_);
            assert(_nodes.count(parentId));
            Node &parent=_nodes[parentId];
            parent.children.reserve(new_nodes.size());
            for(auto &n:new_nodes){
                _nodes.insert(std::make_pair(n.id,n));
                parent.children.push_back(n.id);
            }
        }
        //not thread safe
        size_t size()const{return _nodes.size();}
        std::map<uint32_t,Node> &getNodes(){return _nodes;}
    private:
        std::map<uint32_t,Node> _nodes;
        std::mutex mutex_;

    };

    using vector_sptr=std::shared_ptr< std::vector<uint32_t>>;
    struct ThreadSafeMap{

        void create(uint32_t parent,uint32_t reserved_size){
            std::unique_lock<std::mutex> mlock(mutex_);
            if(!parents_idx.count(parent)){
                parents_idx[parent]=std::make_shared<std::vector<uint32_t>>();
                parents_idx[parent]->reserve(reserved_size);
            }
        }

        void erase(uint32_t parent){
            std::unique_lock<std::mutex> mlock(mutex_);
            assert(parents_idx.count(parent));
            parents_idx.erase(parent);

        }

        vector_sptr operator[](uint32_t parent){
            std::unique_lock<std::mutex> mlock(mutex_);
            assert(parents_idx.count(parent));
            return parents_idx[parent];
        }

        size_t count(uint32_t parent) {
            std::unique_lock<std::mutex> mlock(mutex_);
            return parents_idx.count(parent);
        }
        std::mutex mutex_;
        std::map<uint32_t,vector_sptr> parents_idx;

    };


    Tree TheTree;

    int _descCols,_descType,_descNBytes;
    FeatureInfo _features;
    void assignToClusters(const std::vector<uint32_t> &findices, const std::vector<cv::Mat> &center_features, std::vector<vector_sptr> &assigments, bool omp=false);
    std::vector<cv::Mat>  recomputeCenters(const std::vector<vector_sptr> &assigments, bool omp=false);
    std::size_t vhash(const std::vector<vector_sptr>& v_vec)  ;

    //std::map<uint64,std::vector<uint32_t> > id_assigments;//for each node, its assigment vector

    ThreadSafeMap id_assigments;
    std::vector<std::thread> _Threads;

    const uint32_t maxthreads =100;
    std::atomic<bool> threadRunning[100];//do not  know how to create dinamically :S


    //------------
    void convertIntoVoc(Vocabulary &Voc, std::string dec_name);


    /**
       * Calculates the distance between two descriptors
       * @param a
       * @param b
       * @return distance
       */
    static float distance_float_generic(const cv::Mat &a, const cv::Mat &b);
    static float distance_hamming_generic(const cv::Mat &a, const cv::Mat &b);
    static float distance_hamming_32bytes(const cv::Mat &a, const cv::Mat &b);
    std::function<float(const cv::Mat &a, const cv::Mat &b)> dist_func;
    static inline uint64_t uint64_popcnt(uint64_t v) {
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &   (uint64_t)~(uint64_t)0/15*3);
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
        return (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>  (sizeof(uint64_t) - 1) * 8;
    }
};
}
#endif
