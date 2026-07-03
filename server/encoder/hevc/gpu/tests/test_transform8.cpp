#include "../../transform.h"
#include "../vk_compute.h"
#include <cstdio>
using namespace wivrn::hevc::gpu;
int main(int argc,char**argv){
  vk_compute vk; vk.init();
  const uint32_t nb=8192; const int bd=8;
  auto resb=vk.make_buffer(nb*64*4), levb=vk.make_buffer(nb*64*4), recb=vk.make_buffer(nb*64*4);
  auto pipe=vk.make_pipeline(argv[1],3,12);
  int32_t*rp=(int32_t*)resb.ptr; uint32_t s=999;
  auto rnd=[&](){s=s*1664525u+1013904223u;return s;};
  for(uint32_t i=0;i<nb*64;i++) rp[i]=(int)(rnd()%511u)-255;
  int tot=0;
  for(int qp:{4,12,22,26,32,40,51}){
    struct{uint32_t nb;int qp,bd;}pc{nb,qp,bd};
    vk.run(pipe,{&resb,&levb,&recb},nb,1,1,&pc,sizeof(pc));
    int32_t*gl=(int32_t*)levb.ptr,*gr=(int32_t*)recb.ptr; int fl=0,fr=0;
    for(uint32_t b=0;b<nb;b++){
      int32_t co[64],cl[64],cd[64],cr[64];
      wivrn::hevc::xform::fdct(rp+b*64,co,8,bd);
      wivrn::hevc::xform::quant(co,cl,8,qp,bd);
      wivrn::hevc::xform::dequant(cl,cd,8,qp,bd);
      wivrn::hevc::xform::idct(cd,cr,8,bd);
      for(int i=0;i<64;i++){ if(cl[i]!=gl[b*64+i])fl++; if(cr[i]!=gr[b*64+i])fr++; }
    }
    printf("qp%2d: levels %s (%d)  recon %s (%d)\n",qp,fl?"FAIL":"MATCH",fl,fr?"FAIL":"MATCH",fr);
    tot+=fl+fr;
  }
  printf("\n%s\n",tot?"FAILED":"GPU TRANSFORM8 ROUND-TRIP BIT-EXACT vs CPU");
  return tot?1:0;
}
