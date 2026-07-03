#include "../vk_compute.h"
#include "../../cpu_encoder.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
using namespace wivrn::hevc;
using namespace wivrn::hevc::gpu;

int main(int argc,char**argv){
  int W = argc>2?atoi(argv[2]):256, H=argc>3?atoi(argv[3]):256, qp=argc>4?atoi(argv[4]):26;
  // coded dims (multiple of 64)
  int cw=((W+63)/64)*64, ch=((H+63)/64)*64;
  // build source luma (gradient + sinusoid)
  std::vector<int32_t> srcY((size_t)cw*ch);
  yuv_image img; img.width=cw; img.height=ch;
  img.Y.resize((size_t)cw*ch); img.Cb.assign((size_t)(cw/2)*(ch/2),128); img.Cr.assign((size_t)(cw/2)*(ch/2),128);
  for(int y=0;y<ch;y++)for(int x=0;x<cw;x++){
    double v=128+80*std::sin(x*0.05)*std::cos(y*0.037)+(x-cw/2)*0.1;
    int iv=(int)std::lround(v); iv=iv<0?0:iv>255?255:iv;
    srcY[(size_t)y*cw+x]=iv; img.Y[(size_t)y*cw+x]=(uint16_t)iv;
  }
  // CPU reference: DC-only
  hevc_config cfg; cfg.width=W; cfg.height=H; cfg.qp=qp; cfg.bit_depth=8; cfg.max_tb_log2_size=3; cfg.force_luma_mode=1;
  yuv_image recon;
  encode_intra_frame(cfg, img, &recon);   // recon at coded dims

  // GPU wavefront
  vk_compute vk; vk.init();
  auto sY=vk.make_buffer((size_t)cw*ch*4), rY=vk.make_buffer((size_t)cw*ch*4);
  auto lY=vk.make_buffer((size_t)(cw/8)*(ch/8)*64*4), cbf=vk.make_buffer((size_t)(cw/8)*(ch/8)*4);
  memcpy(sY.ptr, srcY.data(), (size_t)cw*ch*4);
  auto pipe=vk.make_pipeline(argv[1],4,24);
  int bw=cw/8, bh=ch/8;
  struct PC{uint32_t w,h;int qp,bd;uint32_t diag,bx_start;};
  std::vector<vk_compute::step> steps;
  for(int d=0; d<=bw+bh-2; d++){
    int bxs=std::max(0,d-(bh-1)), bxe=std::min(d,bw-1);
    PC pc{(uint32_t)cw,(uint32_t)ch,qp,8,(uint32_t)d,(uint32_t)bxs};
    vk_compute::step s; s.gx=bxe-bxs+1; s.gy=1; s.gz=1;
    s.push.resize(sizeof(pc)); memcpy(s.push.data(),&pc,sizeof(pc));
    steps.push_back(s);
  }
  vk.run_wavefront(pipe, {&sY,&rY,&lY,&cbf}, steps);
  int32_t*g=(int32_t*)rY.ptr;
  int mism=0,maxd=0;
  for(size_t i=0;i<(size_t)cw*ch;i++){ int c=recon.Y[i], gg=g[i]; if(c!=gg){mism++; maxd=std::max(maxd,abs(c-gg));} }
  printf("%dx%d(coded %dx%d) qp%d: luma recon %s (%d mismatch, maxdiff %d)\n",W,H,cw,ch,qp,
         mism?"FAIL":"MATCH", mism, maxd);
  return mism?1:0;
}
