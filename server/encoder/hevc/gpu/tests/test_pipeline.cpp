// Full GPU-math + CPU-CABAC pipeline: GPU reconstruction wavefront -> block
// syntax -> CABAC slice -> writes stream.265 + gpu recon (coded I420).
#include "../vk_compute.h"
#include "../../cpu_encoder.h"
#include "../../cabac_pass.h"
#include "../../param_sets.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
using namespace wivrn::hevc;
using namespace wivrn::hevc::gpu;
static int chroma_qp(int q){ if(q<30)return q; if(q>43)return q-6; static const int t[14]={29,30,31,32,33,33,34,34,35,35,36,36,37,37}; return t[q-30]; }
struct PC{uint32_t w,h;int qp,bd;uint32_t diag,bx_start;};
static std::vector<vk_compute::step> wf(int W,int H,int qp,int bw,int bh){
  std::vector<vk_compute::step> st;
  for(int d=0;d<=bw+bh-2;d++){ int bxs=std::max(0,d-(bh-1)),bxe=std::min(d,bw-1);
    PC pc{(uint32_t)W,(uint32_t)H,qp,8,(uint32_t)d,(uint32_t)bxs};
    vk_compute::step s;s.gx=bxe-bxs+1;s.gy=1;s.gz=1;s.push.resize(sizeof(pc));memcpy(s.push.data(),&pc,sizeof(pc));st.push_back(s);} return st;
}
int main(int argc,char**argv){
  const char*sL=argv[1],*sC=argv[2],*base=argv[3];
  int W=atoi(argv[4]),H=atoi(argv[5]),qp=atoi(argv[6]);
  hevc_config cfg; cfg.width=W;cfg.height=H;cfg.qp=qp;cfg.bit_depth=8;cfg.max_tb_log2_size=3;
  int cw=cfg.coded_width(),ch=cfg.coded_height(),cw2=cw/2,ch2=ch/2,bw=cw/8,bh=ch/8;
  std::vector<int32_t> sY(cw*ch),sCb(cw2*ch2),sCr(cw2*ch2);
  for(int y=0;y<ch;y++)for(int x=0;x<cw;x++){int v=(int)std::lround(128+80*std::sin(x*0.05)*std::cos(y*0.037)+(x-cw/2)*0.1);sY[y*cw+x]=v<0?0:v>255?255:v;}
  for(int y=0;y<ch2;y++)for(int x=0;x<cw2;x++){int b=(int)std::lround(128+50*std::sin(x*0.09)+(y-ch2/2)*0.2);sCb[y*cw2+x]=b<0?0:b>255?255:b;int r=(int)std::lround(128+40*std::cos(y*0.08)-(x-cw2/2)*0.15);sCr[y*cw2+x]=r<0?0:r>255?255:r;}
  vk_compute vk; vk.init();
  block_syntax bs; int nb=bw*bh;
  bs.mode.assign(nb,1); // DC
  bs.cbf_luma.resize(nb);bs.cbf_cb.resize(nb);bs.cbf_cr.resize(nb);
  bs.lev_y.resize((size_t)nb*64);bs.lev_cb.resize((size_t)nb*16);bs.lev_cr.resize((size_t)nb*16);
  std::vector<uint8_t> recY(cw*ch),recCb(cw2*ch2),recCr(cw2*ch2);
  // luma
  { auto sb=vk.make_buffer((size_t)cw*ch*4),rb=vk.make_buffer((size_t)cw*ch*4),lb=vk.make_buffer((size_t)nb*64*4),cb=vk.make_buffer((size_t)nb*4);
    memcpy(sb.ptr,sY.data(),(size_t)cw*ch*4);
    auto p=vk.make_pipeline(sL,4,24); vk.run_wavefront(p,{&sb,&rb,&lb,&cb},wf(cw,ch,qp,bw,bh));
    int32_t*r=(int32_t*)rb.ptr; for(size_t i=0;i<(size_t)cw*ch;i++)recY[i]=(uint8_t)r[i];
    memcpy(bs.lev_y.data(),lb.ptr,(size_t)nb*64*4); uint32_t*c=(uint32_t*)cb.ptr; for(int i=0;i<nb;i++)bs.cbf_luma[i]=(uint8_t)c[i]; }
  auto pC=vk.make_pipeline(sC,4,24); int qc=chroma_qp(qp);
  auto ch_run=[&](std::vector<int32_t>&src,std::vector<uint8_t>&rec,std::vector<int32_t>&lev,std::vector<uint8_t>&cbf){
    auto sb=vk.make_buffer((size_t)cw2*ch2*4),rb=vk.make_buffer((size_t)cw2*ch2*4),lb=vk.make_buffer((size_t)nb*16*4),cb=vk.make_buffer((size_t)nb*4);
    memcpy(sb.ptr,src.data(),(size_t)cw2*ch2*4); vk.run_wavefront(pC,{&sb,&rb,&lb,&cb},wf(cw2,ch2,qc,bw,bh));
    int32_t*r=(int32_t*)rb.ptr; for(size_t i=0;i<(size_t)cw2*ch2;i++)rec[i]=(uint8_t)r[i];
    memcpy(lev.data(),lb.ptr,(size_t)nb*16*4); uint32_t*c=(uint32_t*)cb.ptr; for(int i=0;i<nb;i++)cbf[i]=(uint8_t)c[i];
    vk.destroy_buffer(sb);vk.destroy_buffer(rb);vk.destroy_buffer(lb);vk.destroy_buffer(cb); };
  ch_run(sCb,recCb,bs.lev_cb,bs.cbf_cb); ch_run(sCr,recCr,bs.lev_cr,bs.cbf_cr);

  auto ps=build_parameter_sets(cfg); auto slice=encode_slice_from_syntax(cfg,bs);
  std::vector<uint8_t> stream=ps; stream.insert(stream.end(),slice.begin(),slice.end());
  auto wr=[&](const char*suf,const void*d,size_t n){ std::string p=std::string(base)+suf; FILE*f=fopen(p.c_str(),"wb"); fwrite(d,1,n,f); fclose(f); };
  wr(".265",stream.data(),stream.size());
  { std::vector<uint8_t> buf; buf.insert(buf.end(),recY.begin(),recY.end()); buf.insert(buf.end(),recCb.begin(),recCb.end()); buf.insert(buf.end(),recCr.begin(),recCr.end()); wr("_gpurec.yuv",buf.data(),buf.size()); }
  printf("%dx%d(coded %dx%d) qp%d: stream %zu bytes\n",W,H,cw,ch,qp,stream.size());
  return 0;
}
