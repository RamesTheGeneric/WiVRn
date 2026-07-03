#include "../gpu/gpu_reconstruct.h"
#include "../param_sets.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
using namespace wivrn::hevc;
int main(int argc,char**argv){
  const char*sL=argv[1],*sC=argv[2],*base=argv[3];
  int W=atoi(argv[4]),H=atoi(argv[5]),qp=atoi(argv[6]);
  hevc_config cfg; cfg.width=W;cfg.height=H;cfg.qp=qp;cfg.bit_depth=8;cfg.max_tb_log2_size=3;
  int cw=cfg.coded_width(),ch=cfg.coded_height(),cw2=cw/2,ch2=ch/2;
  std::vector<int32_t> sY(cw*ch),sCb(cw2*ch2),sCr(cw2*ch2);
  for(int y=0;y<ch;y++)for(int x=0;x<cw;x++){int v=(int)std::lround(128+80*std::sin(x*0.05)*std::cos(y*0.037)+(x-cw/2)*0.1);sY[y*cw+x]=v<0?0:v>255?255:v;}
  for(int y=0;y<ch2;y++)for(int x=0;x<cw2;x++){int b=(int)std::lround(128+50*std::sin(x*0.09)+(y-ch2/2)*0.2);sCb[y*cw2+x]=b<0?0:b>255?255:b;int r=(int)std::lround(128+40*std::cos(y*0.08)-(x-cw2/2)*0.15);sCr[y*cw2+x]=r<0?0:r>255?255:r;}
  gpu::reconstructor rec; rec.init_own(sL,sC);
  block_syntax bs;
  std::vector<uint8_t> recY(cw*ch),recCb(cw2*ch2),recCr(cw2*ch2);
  rec.reconstruct(cfg,sY.data(),sCb.data(),sCr.data(),bs,recY.data(),recCb.data(),recCr.data());
  // also reconstruct a 2nd frame to exercise buffer reuse
  rec.reconstruct(cfg,sY.data(),sCb.data(),sCr.data(),bs,recY.data(),recCb.data(),recCr.data());
  auto ps=build_parameter_sets(cfg); auto slice=encode_slice_from_syntax(cfg,bs);
  std::vector<uint8_t> stream=ps; stream.insert(stream.end(),slice.begin(),slice.end());
  auto wr=[&](const char*suf,const void*d,size_t n){std::string p=std::string(base)+suf;FILE*f=fopen(p.c_str(),"wb");fwrite(d,1,n,f);fclose(f);};
  wr(".265",stream.data(),stream.size());
  std::vector<uint8_t> buf; buf.insert(buf.end(),recY.begin(),recY.end());buf.insert(buf.end(),recCb.begin(),recCb.end());buf.insert(buf.end(),recCr.begin(),recCr.end());
  wr("_gpurec.yuv",buf.data(),buf.size());
  printf("%dx%d qp%d: stream %zu bytes\n",W,H,qp,stream.size());
  return 0;
}
