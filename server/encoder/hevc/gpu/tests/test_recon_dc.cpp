#include "../vk_compute.h"
#include "../../cpu_encoder.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
using namespace wivrn::h267;
using namespace wivrn::h267::gpu;

static int chroma_qp(int q){ if(q<30)return q; if(q>43)return q-6; static const int t[14]={29,30,31,32,33,33,34,34,35,35,36,36,37,37}; return t[q-30]; }

struct PC{uint32_t w,h;int qp,bd;uint32_t diag,bx_start;};

// build the per-diagonal wavefront steps for a block grid bw x bh
static std::vector<vk_compute::step> wf(int W,int H,int qp,int bw,int bh){
  std::vector<vk_compute::step> steps;
  for(int d=0; d<=bw+bh-2; d++){
    int bxs=std::max(0,d-(bh-1)), bxe=std::min(d,bw-1);
    PC pc{(uint32_t)W,(uint32_t)H,qp,8,(uint32_t)d,(uint32_t)bxs};
    vk_compute::step s; s.gx=bxe-bxs+1; s.gy=1; s.gz=1;
    s.push.resize(sizeof(pc)); memcpy(s.push.data(),&pc,sizeof(pc)); steps.push_back(s);
  }
  return steps;
}

int main(int argc,char**argv){
  const char* spvL=argv[1]; const char* spvC=argv[2];
  int W=argc>3?atoi(argv[3]):256, H=argc>4?atoi(argv[4]):256, qp=argc>5?atoi(argv[5]):26;
  int cw=((W+63)/64)*64, ch=((H+63)/64)*64, cw2=cw/2, ch2=ch/2;
  yuv_image img; img.width=cw; img.height=ch;
  img.Y.resize((size_t)cw*ch); img.Cb.resize((size_t)cw2*ch2); img.Cr.resize((size_t)cw2*ch2);
  std::vector<int32_t> sY(cw*ch), sCb(cw2*ch2), sCr(cw2*ch2);
  for(int y=0;y<ch;y++)for(int x=0;x<cw;x++){ int v=(int)std::lround(128+80*std::sin(x*0.05)*std::cos(y*0.037)+(x-cw/2)*0.1); v=v<0?0:v>255?255:v; sY[y*cw+x]=v; img.Y[y*cw+x]=v; }
  for(int y=0;y<ch2;y++)for(int x=0;x<cw2;x++){ int b=(int)std::lround(128+50*std::sin(x*0.09)+(y-ch2/2)*0.2); b=b<0?0:b>255?255:b; int r=(int)std::lround(128+40*std::cos(y*0.08)-(x-cw2/2)*0.15); r=r<0?0:r>255?255:r; sCb[y*cw2+x]=b;img.Cb[y*cw2+x]=b; sCr[y*cw2+x]=r;img.Cr[y*cw2+x]=r; }

  hevc_config cfg; cfg.width=W; cfg.height=H; cfg.qp=qp; cfg.bit_depth=8; cfg.max_tb_log2_size=3; cfg.force_luma_mode=1;
  yuv_image rec; encode_intra_frame(cfg,img,&rec);

  vk_compute vk; vk.init();
  int bw=cw/8, bh=ch/8;
  // luma
  auto sYb=vk.make_buffer((size_t)cw*ch*4), rYb=vk.make_buffer((size_t)cw*ch*4);
  auto lYb=vk.make_buffer((size_t)bw*bh*64*4), cYb=vk.make_buffer((size_t)bw*bh*4);
  memcpy(sYb.ptr,sY.data(),(size_t)cw*ch*4);
  auto pL=vk.make_pipeline(spvL,4,24);
  vk.run_wavefront(pL,{&sYb,&rYb,&lYb,&cYb},wf(cw,ch,qp,bw,bh));
  // chroma (bw x bh blocks of 4x4 in cw2 x ch2 plane; blocks-per-row = cw2/4 = bw)
  int qc=chroma_qp(qp);
  auto pC=vk.make_pipeline(spvC,4,24);
  auto run_ch=[&](std::vector<int32_t>&src)->std::vector<int32_t>{
    auto sb=vk.make_buffer((size_t)cw2*ch2*4), rb=vk.make_buffer((size_t)cw2*ch2*4);
    auto lb=vk.make_buffer((size_t)bw*bh*16*4), cb=vk.make_buffer((size_t)bw*bh*4);
    memcpy(sb.ptr,src.data(),(size_t)cw2*ch2*4);
    vk.run_wavefront(pC,{&sb,&rb,&lb,&cb},wf(cw2,ch2,qc,bw,bh));
    std::vector<int32_t> out(cw2*ch2); memcpy(out.data(),rb.ptr,(size_t)cw2*ch2*4);
    vk.destroy_buffer(sb);vk.destroy_buffer(rb);vk.destroy_buffer(lb);vk.destroy_buffer(cb);
    return out;
  };
  auto gCb=run_ch(sCb), gCr=run_ch(sCr);
  int32_t*gY=(int32_t*)rYb.ptr;
  int mY=0,mB=0,mR=0;
  for(size_t i=0;i<(size_t)cw*ch;i++) if(rec.Y[i]!=gY[i])mY++;
  for(size_t i=0;i<(size_t)cw2*ch2;i++){ if(rec.Cb[i]!=gCb[i])mB++; if(rec.Cr[i]!=gCr[i])mR++; }
  printf("%dx%d qp%d: Y %s(%d)  Cb %s(%d)  Cr %s(%d)\n",W,H,qp,mY?"FAIL":"OK",mY,mB?"FAIL":"OK",mB,mR?"FAIL":"OK",mR);
  return (mY||mB||mR)?1:0;
}
