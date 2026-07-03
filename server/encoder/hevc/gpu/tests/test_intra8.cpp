#include "../vk_compute.h"
#include <cstdio>
#include <vector>
using namespace wivrn::hevc::gpu;
// CPU reference predict (verified formulas from cpu_encoder.cpp)
static void cpu_predict(int mode,int cidx,const int*RT,const int*RL,int*out){
  const int N=8,log2n=3;
  if(mode==0){ int tr=RT[N+1],bl=RL[N+1];
    for(int y=0;y<N;y++)for(int x=0;x<N;x++)
      out[y*N+x]=((N-1-x)*RL[1+y]+(x+1)*tr+(N-1-y)*RT[1+x]+(y+1)*bl+N)>>(log2n+1);
  } else { int sum=0; for(int i=0;i<N;i++)sum+=RT[1+i]+RL[1+i]; int dc=(sum+N)>>(log2n+1);
    for(int i=0;i<N*N;i++)out[i]=dc;
    if(cidx==0){ out[0]=(RL[1]+2*dc+RT[1]+2)>>2;
      for(int x=1;x<N;x++)out[x]=(RT[1+x]+3*dc+2)>>2;
      for(int y=1;y<N;y++)out[y*N]=(RL[1+y]+3*dc+2)>>2; }
  }
}
int main(int argc,char**argv){
  vk_compute vk; vk.init();
  const uint32_t nb=8192;
  auto refb=vk.make_buffer(nb*34*4), prb=vk.make_buffer(nb*64*4);
  auto pipe=vk.make_pipeline(argv[1],2,16);
  int32_t*rf=(int32_t*)refb.ptr; uint32_t s=7;
  auto rnd=[&](){s=s*1664525u+1013904223u;return s;};
  for(uint32_t b=0;b<nb;b++)for(int k=0;k<34;k++) rf[b*34+k]=(int)(rnd()%256u);
  int tot=0;
  for(int mode:{0,1})for(int cidx:{0,1}){
    struct{uint32_t nb;int mode,cidx,bd;}pc{nb,mode,cidx,8};
    vk.run(pipe,{&refb,&prb},nb,1,1,&pc,sizeof(pc));
    int32_t*gp=(int32_t*)prb.ptr; int f=0;
    for(uint32_t b=0;b<nb;b++){ int cp[64];
      cpu_predict(mode,cidx,rf+b*34,rf+b*34+17,cp);
      for(int i=0;i<64;i++) if(cp[i]!=gp[b*64+i]) f++;
    }
    printf("mode=%s cidx=%d: %s (%d)\n",mode?"DC":"Planar",cidx,f?"FAIL":"MATCH",f); tot+=f;
  }
  printf("\n%s\n",tot?"FAILED":"GPU INTRA8 (DC/Planar) BIT-EXACT vs CPU");
  return tot?1:0;
}
