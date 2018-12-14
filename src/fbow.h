#ifndef _FBOW_VOCABULARY_H
#define _FBOW_VOCABULARY_H
#include "fbow_exports.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <map>
#include <memory>
#include <bitset>
#ifndef __ANDROID__
#include <immintrin.h>
#endif
#include "cpu.h"
namespace fbow{

//float initialized to zero.
struct FBOW_API _float{
    float var=0;
    inline float operator=(float &f){var=f;return var;}
    inline operator float&() {return var;}
    inline operator float() const{return var;}
};

/**Bag of words
 */
struct FBOW_API fBow:std::map<uint32_t,_float>{

    void toStream(std::ostream &str) const  ;
    void fromStream(std::istream &str)    ;

    //returns a hash identifying this
    uint64_t hash()const;
    //returns the similitude score between to image descriptors using L2 norm
    static double score(const fBow &v1, const fBow &v2);

};


//Bag of words with augmented information. For each word, keeps information about the indices of the elements that have been classified into the word
//it is computed at the desired level
struct FBOW_API fBow2:std::map<uint32_t,std::vector<uint32_t>> {

    void toStream(std::ostream &str) const   ;

    void fromStream(std::istream &str)    ;

    //returns a hash identifying this
    uint64_t hash()const;


};

/**Main class to represent a vocabulary of visual words
 */
class FBOW_API Vocabulary
{
 static inline void * AlignedAlloc(int __alignment,int size){
     assert(__alignment<256);

     unsigned char *ptr= (unsigned  char*)malloc(size + __alignment);

     if( !ptr )  return 0;

     // align the pointer

     size_t lptr=(size_t)ptr;
     int off =lptr%__alignment;
     if (off==0) off=__alignment;

     ptr = ptr+off ; //move to next aligned address
     *(ptr-1)=(unsigned char)off; //save in prev, the offset  to properly remove it
     return ptr;
 }
  
     static inline void AlignedFree(void *ptr){
         if(ptr==nullptr)return;
         unsigned char *uptr=(unsigned char *)ptr;
         unsigned char off= *(uptr-1);
         uptr-=off;
         std::free(uptr);
     }
  
 // using Data_ptr = std::unique_ptr<char[], decltype(&AlignedFree)>;

    friend class VocabularyCreator;

 public:

    Vocabulary(): _data((char*)nullptr,&AlignedFree){}
    Vocabulary(Vocabulary&&) = default;

    //transform the features stored as rows in the returned BagOfWords
    fBow transform(const cv::Mat &features);
    void transform(const cv::Mat &features, int level,fBow &result,fBow2&result2);


    //loads/saves from a file
    void readFromFile(const std::string &filepath);
    void saveToFile(const std::string &filepath);
    ///save/load to binary streams
    void toStream(std::ostream &str) const;
    void fromStream(std::istream &str);
    //returns the descriptor type (CV_8UC1, CV_32FC1  )
    uint32_t getDescType()const{return _params._desc_type;}
    //returns desc size in bytes or 0 if not set
    uint32_t getDescSize()const{return _params._desc_size;}
    //returns the descriptor name
    std::string getDescName() const{ return _params._desc_name_;}
    //returns the branching factor (number of children per node)
    uint32_t getK()const{return _params._m_k;}
    //indicates whether this object is valid
    bool isValid()const{return _data.get()!=nullptr;}
    //total number of blocks
    size_t size()const{return _params._nblocks;}
    //removes all data
    void clear();
    //returns a hash value idinfying the vocabulary
    uint64_t hash()const;

private:
     void  setParams(  int aligment,int k,int desc_type,int desc_size, int nblocks,std::string desc_name) ;
    struct params{
        char _desc_name_[50];//descriptor name. May be empty
        uint32_t _aligment=0,_nblocks=0 ;//memory aligment and total number of blocks
        uint64_t _desc_size_bytes_wp=0;//size of the descriptor(includes padding)
        uint64_t _block_size_bytes_wp=0;//size of a block   (includes padding)
        uint64_t _feature_off_start=0;//within a block, where the features start
        uint64_t _child_off_start=0;//within a block,where the children offset part starts
        uint64_t _total_size=0;
        int32_t _desc_type=0,_desc_size=0;//original descriptor types and sizes (without padding)
        uint32_t _m_k=0;//number of children per node
    };
    params _params;
    std::unique_ptr<char[], decltype(&AlignedFree)> _data;


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
    inline Block getBlock(uint32_t b) { assert(_data.get() != nullptr); assert(b < _params._nblocks); return Block(_data.get() + b * _params._block_size_bytes_wp, _params._desc_size, _params._desc_size_bytes_wp, _params._feature_off_start, _params._child_off_start); }
    //given a block already create with getBlock, moves it to point to block b
    inline void setBlock(uint32_t b, Block &block) { block._blockstart = _data.get() + b * _params._block_size_bytes_wp; }

    //information about the cpu so that mmx,sse or avx extensions can be employed
    std::shared_ptr<cpu> cpu_info;


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
        virtual ~Lx(){if (feature!=0)AlignedFree(feature);}
        void setParams(int desc_size, int block_desc_size_bytes_wp){
            assert(block_desc_size_bytes_wp%aligment==0);
            _desc_size=desc_size;
            _block_desc_size_bytes_wp=block_desc_size_bytes_wp;
            assert(_block_desc_size_bytes_wp%sizeof(register_type )==0);
            _nwords=_block_desc_size_bytes_wp/sizeof(register_type );//number of aligned words
            feature=static_cast<register_type*> (AlignedAlloc(aligment,_nwords*sizeof(register_type )));
           memset(feature,0,_nwords*sizeof(register_type ));
        }
        inline void startwithfeature(const register_type *feat_ptr){memcpy(feature,feat_ptr,_desc_size);}
        virtual distType computeDist(register_type *fptr)=0;

    };


    struct L2_generic:public Lx<float,float,4>{
        virtual ~L2_generic(){ }
        inline float computeDist(float *fptr){
            float d=0;
            for(int f=0;f<_nwords;f++)  d+=  (feature[f]-fptr[f])*(feature[f]-fptr[f]);
            return d;
        }
    };
#ifdef __ANDROID__
    //fake elements to allow compilation
    struct L2_avx_generic:public Lx<uint64_t,float,32>{inline float computeDist(uint64_t *ptr){return std::numeric_limits<float>::max();}};
    struct L2_se3_generic:public Lx<uint64_t,float,32>{inline float computeDist(uint64_t *ptr){return std::numeric_limits<float>::max();}};
    struct L2_sse3_16w:public Lx<uint64_t,float,32>{inline float computeDist(uint64_t *ptr){return std::numeric_limits<float>::max();}};
    struct L2_avx_8w:public Lx<uint64_t,float,32>{inline float computeDist(uint64_t *ptr){return std::numeric_limits<float>::max();}};




#else
    struct L2_avx_generic:public Lx<__m256,float,32>{
        virtual ~L2_avx_generic(){}
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

 #endif

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
     fBow  _transform(const cv::Mat &features){
         Computer comp;
         comp.setParams(_params._desc_size,_params._desc_size_bytes_wp);
         using DType=typename Computer::DType;//distance type
         using TData=typename Computer::TData;//data type

         fBow  result;
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
                 if ( bn_info->isleaf()){//if the node is leaf get word id and weight
                      result[bn_info->getId()]+=bn_info->weight;
                  }
                 else setBlock(bn_info->getId(),c_block);//go to its children
             }while( !bn_info->isleaf() && bn_info->getId()!=0);
         }
         return result;
     }
     template<typename Computer>
     void  _transform2(const cv::Mat &features,uint32_t storeLevel,fBow &r1,fBow2 &r2){
         Computer comp;
              comp.setParams(_params._desc_size,_params._desc_size_bytes_wp);
              using DType=typename Computer::DType;//distance type
              using TData=typename Computer::TData;//data type

              r1.clear();
              r2.clear();
              std::pair<DType,uint32_t> best_dist_idx(std::numeric_limits<uint32_t>::max(),0);//minimum distance found
              block_node_info *bn_info;
              int nbits=ceil(log2(_params._m_k));
              for(int cur_feature=0;cur_feature<features.rows;cur_feature++){
                  comp.startwithfeature(features.ptr<TData>(cur_feature));
                  //ensure feature is in a
                  Block c_block=getBlock(0);
                  uint32_t level=0;//current level of recursion
                  uint32_t curNode=0;//id of the current node of the tree
                  //copy to another structure and add padding with zeros
                  do{
                      //given the current block, finds the node with minimum distance
                      best_dist_idx.first=std::numeric_limits<uint32_t>::max();
                      for(int cur_node=0;cur_node<c_block.getN();cur_node++)
                      {
                          DType d= comp.computeDist(c_block.getFeature<TData>(cur_node));
                          if (d<best_dist_idx.first) best_dist_idx=std::make_pair(d,cur_node);
                      }
                      if( level==storeLevel)//if reached level,save
                          r2[curNode].push_back( cur_feature);

                      bn_info=c_block.getBlockNodeInfo(best_dist_idx.second);
                      //if the node is leaf get weight,else go to its children
                      if ( bn_info->isleaf()){
                          r1[bn_info->getId()]+=bn_info->weight;
                          if( level<storeLevel)//store level not reached, save now
                              r2[curNode].push_back( cur_feature);
                        break;
                      }
                      else setBlock(bn_info->getId(),c_block);//go to its children
                      curNode= curNode<<nbits;
                      curNode|=best_dist_idx.second;
                      level++;
                  }while( !bn_info->isleaf() && bn_info->getId()!=0);
              }
      }

};


}
#endif
