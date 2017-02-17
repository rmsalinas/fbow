#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"
#include <opencv2/core/core.hpp>
#include "fbow.h"
#include <set>
using namespace std;
using ORBVocabulary=DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;

namespace fbow{
 class VocabularyCreator{
public:
     struct ninfo{
         ninfo(){}
         ninfo(uint32_t Block,ORBVocabulary::Node *Node):block(Block),node(Node){}
         int64_t block=-1;
         ORBVocabulary::Node *node=0;
     };


     static void convert(ORBVocabulary &voc,fbow::Vocabulary &out_voc){
         uint32_t nonLeafNodes=0;
         std::map<uint32_t,ninfo> nodeid_info;
         for(int i=0;i<voc.m_nodes.size();i++){
             auto &node=voc.m_nodes[i];
             if(!node.isLeaf()) nodeid_info.insert(std::make_pair(node.id,ninfo(nonLeafNodes++,&node)));
             else nodeid_info.insert(std::make_pair(node.id,ninfo(-1,&node)));
         }

         out_voc.setParams(8,voc.m_k,CV_8UC1,32,nonLeafNodes,"orb");
         cerr<<"creating     size="<<out_voc._params._total_size/(1024*1024)<<"Mb "<<out_voc._params._total_size<<" bytes"<<endl;

         for(int i=0;i<voc.m_nodes.size();i++){
             ORBVocabulary::Node &node= voc.m_nodes[i];
             if(!node.isLeaf()) {
                 auto &n_info=nodeid_info[node.id];
                 fbow::Vocabulary::Block binfo=out_voc.getBlock(n_info.block);
                 binfo.setN(node.children.size());
                 binfo.setParentId(node.id);
                 bool areAllChildrenLeaf=true;
                 std::sort(node.children.begin(),node.children.end());
                 for(int c=0;c<node.children.size();c++){
                     auto &child_info=nodeid_info[node.children[c]];
                     binfo.setFeature(c, child_info.node->descriptor);
                     if (child_info.node->isLeaf())
                         binfo.getBlockNodeInfo(c)->setLeaf(child_info.node->word_id,child_info.node->weight);
                     else {
                         areAllChildrenLeaf=false;
                         binfo.getBlockNodeInfo(c)->setNonLeaf(child_info.block);
                     }
                 }
                 binfo.setLeaf(areAllChildrenLeaf);
             }
         }
     }
 };
}

int main(int argc,char **argv){
    if (argc!=3){cerr<<"Usage voc.txt out.fbow"<<endl;return -1;}
    ORBVocabulary voc;
    cout<<"loading dbow2 voc"<<endl;
    voc.loadFromTextFile(argv[1]);
    cout<<"done"<<endl;
    fbow::Vocabulary fvoc;
    fbow::VocabularyCreator::convert(voc,fvoc);
    fvoc.saveToFile(argv[2]);
}
