#ifndef _FBOW_VOCABULARY_H
#define _FBOW_VOCABULARY_H
#include "exports.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <map>
#include <memory>
#include <bitset>
#include "cpu_x86.h"
namespace fbow{

//float initialized to zero.
struct _float{
    float var=0;
    inline float operator=(float &f){var=f;return var;}
    inline operator float&() {return var;}
    inline operator float() const{return var;}
};

/**Bag of words
 */
struct fBow:std::map<uint32_t,_float>{

    void toStream(std::ostream &str) const   {
        uint32_t _size=size();
        str.write((char*)&_size,sizeof(_size));
        for(std::pair<uint32_t,_float> e:*this)
            str.write((char*)&e,sizeof(e));
    }
    void fromStream(std::istream &str)    {
        clear();
        uint32_t _size;
        str.read((char*)&_size,sizeof(_size));
        for(uint32_t i=0;i<_size;i++){
            std::pair<uint32_t,_float> e;
            str.read((char*)&e,sizeof(e));
            insert(e);
        }
    }

    //returns a hash identifying this
    uint64_t hash()const;
    //returns the similitude score between to image descriptors using L2 norm
    static double score(const fBow &v1, const fBow &v2);

};



/**Main class to represent a vocabulary of visual words
 */
class FBOW_API Vocabulary
{
    friend class VocabularyCreator;
 public:

    ~Vocabulary();

    //transform the features stored as rows in the returned BagOfWords
    fBow transform(const cv::Mat &features)throw(std::exception);


    //loads/saves from a file
    void readFromFile(const std::string &filepath)throw(std::exception);    
    void saveToFile(const std::string &filepath)throw(std::exception);
    ///save/load to binary streams
    void toStream(std::ostream &str) const;
    void fromStream(std::istream &str)throw(std::exception);
    //returns the descriptor type (CV_8UC1, CV_32FC1  )
    uint32_t getDescType()const{return _params._desc_type;}
    //returns desc size in bytes or 0 if not set
    uint32_t getDescSize()const{return _params._desc_size;}
    //returns the descriptor name
    std::string getDescName() const{ return _params._desc_name_;}
    //returns the branching factor (number of children per node)
    uint32_t getK()const{return _params._m_k;}
    //indicates whether this object is valid
    bool isValid()const{return _data!=0;}
    //total number of blocks
    size_t size()const{return _params._nblocks;}
    //removes all data
    void clear();
    //returns a hash value idinfying the vocabulary
    uint64_t hash()const;

private:
     void  setParams(  int aligment,int k,int desc_type,int desc_size, int nblocks,std::string desc_name)throw(std::runtime_error);
    struct params{
        char _desc_name_[50]="";//descriptor name. May be empty
        uint32_t _aligment,_nblocks=0 ;//memory aligment and total number of blocks
        uint64_t _desc_size_bytes_wp=0;//size of the descriptor(includes padding)
        uint64_t _block_size_bytes_wp=0;//size of a block   (includes padding)
        uint64_t _feature_off_start=0;//within a block, where the features start
        uint64_t _child_off_start=0;//within a block,where the children offset part starts
        uint64_t _total_size=0;
        int32_t _desc_type=0,_desc_size=0;//original descriptor types and sizes (without padding)
        uint32_t _m_k=0;//number of children per node
    };
    params _params;
    char * _data=0;//pointer to data

    //structure represeting a information about node in a block
    struct block_node_info{
        uint32_t id_or_childblock; //if id ,msb is 1.
        float weight;
        inline bool isleaf()const{return ( id_or_childblock& 0x80000000);}

        //if not leaf, returns the block where the children are
        //if leaf, returns the index of the feature it represents. In case of bagofwords it must be a invalid value
        inline uint32_t getId()const{return ( id_or_childblock&0x7FFFFFFF);}

        //sets as leaf, and sets the index of the feature it represents and its weight
        inline void setLeaf(uint32_t id,float Weight){
            assert(!(id & 0x80000000));//check msb is zero
            id_or_childblock=id;
            id_or_childblock|=0x80000000;//set the msb to one to distinguish from non leaf
            //now,set the weight too
            weight=Weight;
        }
        //sets as non leaf and sets the id of the block where the chilren are
        inline void setNonLeaf(uint32_t id){
            //ensure the msb is 0
            assert( !(id & 0x80000000));//32 bits 100000000...0.check msb is not set
            id_or_childblock=id;
        }
    };


    //a block represent all the child nodes of a parent, with its features and also information about where the child of these are in the data structure
    //a block structure is as follow: N|isLeaf|BlockParentId|p|F0...FN|C0W0 ... CNWN..
    //N :16 bits : number of nodes in this block. Must be <=branching factor k. If N<k, then the block has empty spaces since block size is fixed
    //isLeaf:16 bit inicating if all nodes in this block are leaf or not
    //BlockParentId:31: id of the parent
    //p :possible offset so that Fi is aligned
    //Fi feature of the node i. it is aligned and padding added to the end so that F(i+1) is also aligned
    //CiWi are the so called block_node_info (see structure up)
    //Ci : either if the node is leaf (msb is set to 1) or not. If not leaf, the remaining 31 bits is the block where its children are. Else, it is the index of the feature that it represent
    //Wi: float value empkoyed to know the weight of a leaf node (employed in cases of bagofwords)
    struct Block{
        Block(char * bsptr,uint64_t ds,uint64_t ds_wp,uint64_t fo,uint64_t co):_blockstart(bsptr),_desc_size_bytes(ds),_desc_size_bytes_wp(ds_wp),_feature_off_start(fo),_child_off_start(co){}
        Block(uint64_t ds,uint64_t ds_wp,uint64_t fo,uint64_t co):_desc_size_bytes(ds),_desc_size_bytes_wp(ds_wp),_feature_off_start(fo),_child_off_start(co){}

        inline  uint16_t getN()const{return (*((uint16_t*)(_blockstart)));}
        inline  void setN(uint16_t n){ *((uint16_t*)(_blockstart))=n;}

        inline bool isLeaf()const{return *((uint16_t*)(_blockstart)+1);}
        inline void setLeaf(bool v)const{*((uint16_t*)(_blockstart)+1)=1;}

        inline void setParentId(uint32_t pid){*(((uint32_t*)(_blockstart))+1)=pid;}
        inline uint32_t  getParentId(){ return *(((uint32_t*)(_blockstart))+1);}

        inline  block_node_info * getBlockNodeInfo(int i){  return (block_node_info *)(_blockstart+_child_off_start+i*sizeof(block_node_info)); }
        inline  void setFeature(int i,const cv::Mat &feature){memcpy( _blockstart+_feature_off_start+i*_desc_size_bytes_wp,feature.ptr<char>(0),feature.elemSize1()*feature.cols); }
        inline  void getFeature(int i,cv::Mat  feature){    memcpy( feature.ptr<char>(0), _blockstart+_feature_off_start+i*_desc_size_bytes,_desc_size_bytes ); }
        template<typename T> inline  T*getFeature(int i){return (T*) (_blockstart+_feature_off_start+i*_desc_size_bytes_wp);}
        char *_blockstart;
        uint64_t _desc_size_bytes=0;//size of the descriptor(without padding)
        uint64_t _desc_size_bytes_wp=0;//size of the descriptor(includding padding)
        uint64_t _feature_off_start=0;
        uint64_t _child_off_start=0;//into the block,where the children offset part starts
    };


    //returns a block structure pointing at block b
    inline Block getBlock(uint32_t b){assert( _data!=0);assert(b<_params._nblocks); return Block( _data+ b*_params._block_size_bytes_wp,_params._desc_size, _params._desc_size_bytes_wp,_params._feature_off_start, _params._child_off_start);}
    //given a block already create with getBlock, moves it to point to block b
    inline void setBlock(uint32_t b,Block &block){ block._blockstart= _data+ b*_params._block_size_bytes_wp;}

    //information about the cpu so that mmx,sse or avx extensions can be employed
    std::shared_ptr<cpu_x86> cpu_info;


    ////////////////////////////////////////////////////////////
    //base class for computing distances between feature vectors
    template<typename register_type,typename distType, int aligment>
    class Lx{
    public:
        typedef distType DType;
        typedef register_type TData;
    protected:

        int _nwords,_aligment,_desc_size;
        int _block_desc_size_bytes_wp;
        register_type *feature=0;
    public:
         ~Lx(){if (feature!=0)free(feature);}
        void setParams(int desc_size, int block_desc_size_bytes_wp){
            assert(block_desc_size_bytes_wp%aligment==0);
            _desc_size=desc_size;
            _block_desc_size_bytes_wp=block_desc_size_bytes_wp;
            assert(_block_desc_size_bytes_wp%sizeof(register_type )==0);
            _nwords=_block_desc_size_bytes_wp/sizeof(register_type );//number of aligned words
            
#if _WIN32
            feature = (register_type*)_aligned_malloc(_nwords * sizeof(register_type), aligment);
#else
            feature = (register_type*)aligned_alloc(aligment, _nwords * sizeof(register_type));
#endif
           memset(feature,0,_nwords*sizeof(register_type ));
        }
        inline void startwithfeature(const register_type *feat_ptr){memcpy(feature,feat_ptr,_desc_size);}
        virtual distType computeDist(register_type *fptr)=0;

    };

    struct L2_generic:public Lx<float,float,4>{
         ~L2_generic(){ }
        inline float computeDist(float *fptr){
            float d=0;
            for(int f=0;f<_nwords;f++)  d+=  (feature[f]-fptr[f])*(feature[f]-fptr[f]);
            return d;
        }
    };

    struct L2_avx_generic:public Lx<__m256,float,32>{
        inline float computeDist(__m256 *ptr){
             __m256 sum=_mm256_setzero_ps(), sub_mult;
            //substract, multiply and accumulate
            for(int i=0;i<_nwords;i++){
                sub_mult=_mm256_sub_ps(feature[i],ptr[i]);
                sub_mult=_mm256_mul_ps(sub_mult,sub_mult);
                sum=_mm256_add_ps(sum,sub_mult);
            }
            sum=_mm256_hadd_ps(sum,sum);
            sum=_mm256_hadd_ps(sum,sum);
            float *sum_ptr=(float*)&sum;
            return  sum_ptr[0]+sum_ptr[4];
        }
    };

    struct L2_se3_generic:public Lx<__m128,float,16>{
        inline float computeDist(__m128 *ptr){
             __m128 sum=_mm_setzero_ps(), sub_mult;
            //substract, multiply and accumulate
            for(int i=0;i<_nwords;i++){
                sub_mult=_mm_sub_ps(feature[i],ptr[i]);
                sub_mult=_mm_mul_ps(sub_mult,sub_mult);
                sum=_mm_add_ps(sum,sub_mult);
            }
            sum=_mm_hadd_ps(sum,sum);
            sum=_mm_hadd_ps(sum,sum);
            float *sum_ptr=(float*)&sum;
            return  sum_ptr[0] ;
        }
    };
    struct L2_sse3_16w:public Lx<__m128,float,16> {

        inline float computeDist(__m128 *ptr){
             __m128 sum=_mm_setzero_ps(), sub_mult;
            //substract, multiply and accumulate
            for(int i=0;i<16;i++){
                sub_mult=_mm_sub_ps(feature[i],ptr[i]);
                sub_mult=_mm_mul_ps(sub_mult,sub_mult);
                sum=_mm_add_ps(sum,sub_mult);
            }
            sum=_mm_hadd_ps(sum,sum);
            sum=_mm_hadd_ps(sum,sum);
            float *sum_ptr=(float*)&sum;
            return  sum_ptr[0] ;
        }
    };
    //specific for surf in avx
    struct L2_avx_8w:public Lx<__m256,float,32> {

        inline float computeDist(__m256 *ptr){
             __m256 sum=_mm256_setzero_ps(), sub_mult;
            //substract, multiply and accumulate

            for(int i=0;i<8;i++){
                sub_mult=_mm256_sub_ps(feature[i],ptr[i]);
                sub_mult=_mm256_mul_ps(sub_mult,sub_mult);
                sum=_mm256_add_ps(sum,sub_mult);
            }

            sum=_mm256_hadd_ps(sum,sum);
            sum=_mm256_hadd_ps(sum,sum);
            float *sum_ptr=(float*)&sum;
            return  sum_ptr[0]+sum_ptr[4];
        }


    };



    //generic hamming distance calculator
     struct  L1_x64:public Lx<uint64_t,uint64_t,8>{
         inline uint64_t computeDist(uint64_t *feat_ptr){
             uint64_t result = 0;
             for(int i = 0; i < _nwords; ++i ) result += std::bitset<64>(feat_ptr[i] ^ feature[i]).count();
             return result;
         }
     };

     struct  L1_x32:public Lx<uint32_t,uint32_t,8>{
         inline uint32_t computeDist(uint32_t *feat_ptr){
             uint32_t result = 0;
             for(int i = 0; i < _nwords; ++i ) result +=  std::bitset<32>(feat_ptr[i] ^ feature[i]).count();
             return result;
         }

     };

     //for orb
     struct L1_32bytes:public Lx<uint64_t,uint64_t,8>{
         inline uint64_t computeDist(uint64_t *feat_ptr){
              return uint64_popcnt(feat_ptr[0]^feature[0])+ uint64_popcnt(feat_ptr[1]^feature[1])+
                      uint64_popcnt(feat_ptr[2]^feature[2])+uint64_popcnt(feat_ptr[3]^feature[3]);
         }
         inline uint64_t uint64_popcnt(uint64_t n) {
             return std::bitset<64>(n).count();
         }

     };
     //for akaze
     struct L1_61bytes:public Lx<uint64_t,uint64_t,8>{
         inline uint64_t computeDist(uint64_t *feat_ptr){

              return uint64_popcnt(feat_ptr[0]^feature[0])+ uint64_popcnt(feat_ptr[1]^feature[1])+
                      uint64_popcnt(feat_ptr[2]^feature[2])+uint64_popcnt(feat_ptr[3]^feature[3])+
                      uint64_popcnt(feat_ptr[4]^feature[4])+uint64_popcnt(feat_ptr[5]^feature[5])+
                      uint64_popcnt(feat_ptr[6]^feature[6])+uint64_popcnt(feat_ptr[7]^feature[7]);
         }
         inline uint64_t uint64_popcnt(uint64_t n) {
             return std::bitset<64>(n).count();
         }
     };


    template<typename Computer>
    fBow _transform(const cv::Mat &features)throw(std::exception){
        Computer comp;
        comp.setParams(_params._desc_size,_params._desc_size_bytes_wp);
        using DType=typename Computer::DType;//distance type
        using TData=typename Computer::TData;//data type

        fBow result;
        std::pair<DType,uint32_t> best_dist_idx(std::numeric_limits<uint32_t>::max(),0);//minimum distance found
        block_node_info *bn_info;
        for(int cur_feature=0;cur_feature<features.rows;cur_feature++){
            comp.startwithfeature(features.ptr<TData>(cur_feature));
            //ensure feature is in a
            Block c_block=getBlock(0);
              //copy to another structure and add padding with zeros
            do{
                //given the current block, finds the node with minimum distance
                best_dist_idx.first=std::numeric_limits<uint32_t>::max();
                for(int cur_node=0;cur_node<c_block.getN();cur_node++)
                {
                    DType d= comp.computeDist(c_block.getFeature<TData>(cur_node));
                    if (d<best_dist_idx.first) best_dist_idx=std::make_pair(d,cur_node);
                }
                bn_info=c_block.getBlockNodeInfo(best_dist_idx.second);
                //if the node is leaf get word id and weight,else go to its children
                if ( bn_info->isleaf())
                    result[bn_info->getId()]+=bn_info->weight;//if the node is leaf get word id and weight
                else setBlock(bn_info->getId(),c_block);//go to its children
            }while( !bn_info->isleaf());
        }
        return result;
    }

};


}
#endif
