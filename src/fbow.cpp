#include "fbow.h"
#include <fstream>
#include <cstring>
#include <limits>
#include <cstdint>
#include <algorithm>

namespace fbow{


Vocabulary::~Vocabulary(){
    if (_data!=0) AlignedFree( _data);
}


void Vocabulary::setParams(int aligment, int k, int desc_type, int desc_size, int nblocks, std::string desc_name)throw(std::runtime_error){
    auto ns= desc_name.size()<static_cast<size_t>(49)?desc_name.size():128;
    desc_name.resize(ns);

    std::strcpy(_params._desc_name_,desc_name.c_str());
    _params._aligment=aligment;
    _params._m_k= k;
    _params._desc_type=desc_type;
    _params._desc_size=desc_size;
    _params._nblocks=nblocks;


    uint64_t _desc_size_bytes_al=0;
    uint64_t _block_size_bytes_al=0;

    //consider possible aligment of each descriptor adding offsets at the end
    _params._desc_size_bytes_wp=_params._desc_size;
    _desc_size_bytes_al= _params._desc_size_bytes_wp/ _params._aligment;
    if( _params._desc_size_bytes_wp% _params._aligment!=0)   _desc_size_bytes_al++;
    _params._desc_size_bytes_wp= _desc_size_bytes_al* _params._aligment;


    int foffnbytes_alg=sizeof(uint64_t)/_params._aligment;
    if(sizeof(uint64_t)%_params._aligment!=0) foffnbytes_alg++;
    _params._feature_off_start=foffnbytes_alg*_params._aligment;
    _params._child_off_start=_params._feature_off_start+_params._m_k*_params._desc_size_bytes_wp ;//where do children information start from the start of the block

    //block: nvalid|f0 f1 .. fn|ni0 ni1 ..nin
    _params._block_size_bytes_wp=_params._feature_off_start+  _params._m_k * ( _params._desc_size_bytes_wp + sizeof(Vocabulary::block_node_info));
    _block_size_bytes_al=_params._block_size_bytes_wp/_params._aligment;
    if (_params._block_size_bytes_wp%_params._aligment!=0) _block_size_bytes_al++;
    _params._block_size_bytes_wp= _block_size_bytes_al*_params._aligment;

    //give memory
    _params._total_size=_params._block_size_bytes_wp*_params._nblocks;
    _data=(char*)AlignedAlloc(_params._aligment,_params._total_size);
    memset( _data,0,_params._total_size);

}


fBow Vocabulary::transform(const cv::Mat &features)
{
    if (features.rows==0) throw std::runtime_error("Vocabulary::transform No input data");
    if (features.type()!=_params._desc_type) throw std::runtime_error("Vocabulary::transform features are of different type than vocabulary");
    if (features.cols *  features.elemSize() !=size_t(_params._desc_size)) throw std::runtime_error("Vocabulary::transform features are of different size than the vocabulary ones");

    //get host info to decide the version to execute
    if (!cpu_info){
        cpu_info=std::make_shared<cpu>();
        cpu_info->detect_host();
    }
    fBow result;
    //decide the version to employ according to the type of features, aligment and cpu capabilities
    if (_params._desc_type==CV_8UC1){
        //orb
        if (cpu_info->HW_x64){
            if (_params._desc_size==32)
				result=_transform<L1_32bytes>(features);
            //full akaze
            else if( _params._desc_size==61 && _params._aligment%8==0)
				result=_transform<L1_61bytes>(features);
            //generic
            else
				result=_transform<L1_x64>(features );
        }
        else  result=  _transform<L1_x32>(features );
    }
    else if(features.type()==CV_32FC1){
        if( cpu_info->isSafeAVX() && _params._aligment%32==0){ //AVX version
            if ( _params._desc_size==256) result= _transform<L2_avx_8w>(features);//specific for surf 256 bytes
            else result= _transform<L2_avx_generic>(features);//any other
        }
        if( cpu_info->isSafeSSE() && _params._aligment%16==0){//SSE version
            if ( _params._desc_size==256) result= _transform<L2_sse3_16w>(features);//specific for surf 256 bytes
            else result=_transform<L2_se3_generic>(features);//any other
        }
        //generic version
        result=_transform<L2_generic>(features);
    }
    else throw std::runtime_error("Vocabulary::transform invalid feature type. Should be CV_8UC1 or CV_32FC1");

    ///now, normalize
    //L2
    double norm=0;
    for(auto  e:result) norm += e.second * e.second;

    if(norm > 0.0)
    {
        double inv_norm = 1./sqrt(norm);
        for(auto  &e:result) e.second*=inv_norm ;
    }
    return result;
}




void Vocabulary::clear()
{
    if (_data!=0) AlignedFree(_data);
    _data=0;
    memset(&_params,0,sizeof(_params));
    _params._desc_name_[0]='\0';
}


//loads/saves from a file
void Vocabulary::readFromFile(const std::string &filepath){
    std::ifstream file(filepath,std::ios::binary);
    if (!file) throw std::runtime_error("Vocabulary::readFromFile could not open:"+filepath);
    fromStream(file);
}

void Vocabulary::saveToFile(const std::string &filepath){
	std::ofstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Vocabulary::saveToFile could not open:"+filepath);
    toStream(file);

}

///save/load to binary streams
void Vocabulary::toStream(std::ostream &str)const{
    //magic number
    uint64_t sig=55824124;
    str.write((char*)&sig,sizeof(sig));
    //save string
    str.write((char*)&_params,sizeof(params));
    str.write(_data,_params._total_size);
}

void Vocabulary::fromStream(std::istream &str)
{
    if (_data!=0) AlignedFree (_data);
    uint64_t sig;
    str.read((char*)&sig,sizeof(sig));
    if (sig!=55824124) throw std::runtime_error("Vocabulary::fromStream invalid signature");
    //read string
    str.read((char*)&_params,sizeof(params));
    _data=(char*)AlignedAlloc(_params._aligment,_params._total_size);
    if (_data==0) throw std::runtime_error("Vocabulary::fromStream Could not allocate data");
    str.read(_data,_params._total_size);
}

double fBow::score (const  fBow &v1,const fBow &v2){


    fBow::const_iterator v1_it, v2_it;
    const fBow::const_iterator v1_end = v1.end();
    const fBow::const_iterator v2_end = v2.end();

    v1_it = v1.begin();
    v2_it = v2.begin();

    double score = 0;

    while(v1_it != v1_end && v2_it != v2_end)
    {
        const auto& vi = v1_it->second;
        const auto& wi = v2_it->second;

        if(v1_it->first == v2_it->first)
        {
            score += vi * wi;

            // move v1 and v2 forward
            ++v1_it;
            ++v2_it;
        }
        else if(v1_it->first < v2_it->first)
        {
            // move v1 forward
//            v1_it = v1.lower_bound(v2_it->first);
            while(v1_it!=v1_end&& v1_it->first<v2_it->first)
                ++v1_it;
        }
        else
        {
            // move v2 forward
//            v2_it = v2.lower_bound(v1_it->first);
            while(v2_it!=v2_end && v2_it->first<v1_it->first)
                ++v2_it;

            // v2_it = (first element >= v1_it.id)
        }
    }

    // ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
    //		for all i | v_i != 0 and w_i != 0 )
    // (Nister, 2006)
    if(score >= 1) // rounding errors
        score = 1.0;
    else
        score = 1.0 - sqrt(1.0 - score); // [0..1]

    return score;
}
uint64_t fBow::hash()const{
    uint64_t seed = 0;
    for(auto e:*this)
        seed^= e.first +  int(e.second*1000)+ 0x9e3779b9 + (seed << 6) + (seed >> 2);

    return seed;

}

uint64_t Vocabulary::hash()const{

    uint64_t seed = 0;
    for(uint64_t i=0;i<_params._total_size;i++)
        seed^= _data[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}
}
