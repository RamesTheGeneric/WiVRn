// Validate reconstructor::encode_frame (the integrated GPU recon+CABAC path the
// server uses): encode_frame -> assemble -> ffmpeg decode must equal a reference
// reconstruction pixel-exact.
//
// Build/run (from server/encoder/hevc/):
//   g++ -std=c++20 -I. gpu/tests/test_encode_frame.cpp gpu/gpu_reconstruct.cpp \
//       cabac_pass.cpp residual_coding.cpp cabac_reference.cpp param_sets.cpp bitwriter.cpp \
//       -lvulkan -o /tmp/test_encode_frame
//   /tmp/test_encode_frame /tmp/rl.spv /tmp/rc.spv /tmp/cabac.spv
#include "../../bitwriter.h"
#include "../../cabac_pass.h"
#include "../../param_sets.h"
#include "../gpu_reconstruct.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace wivrn::h267;
using namespace wivrn::h267::gpu;

int main(int argc, char ** argv)
{
	const char * rl = argv[1], *rc = argv[2], *cabac = argv[3];
	int fails = 0;
	struct tc { int w, h, qp, R; };
	for (auto t: {tc{256, 256, 26, 2}, tc{384, 320, 20, 1}, tc{320, 448, 32, 3}})
	{
		hevc_config cfg;
		cfg.width = t.w; cfg.height = t.h; cfg.qp = t.qp; cfg.bit_depth = 8; cfg.max_tb_log2_size = 3;
		const int cw = cfg.coded_width(), ch = cfg.coded_height();
		const int ew = t.w, eh = t.h;
		const uint32_t nx = cfg.ctbs_x();

		std::vector<uint8_t> luma((size_t)ew * eh), chroma((size_t)ew * eh / 2);
		for (int y = 0; y < eh; ++y)
			for (int x = 0; x < ew; ++x)
				luma[(size_t)y * ew + x] = (uint8_t)std::min(255, 30 + (x * 170 / ew) + ((x ^ y) & 27));
		for (int y = 0; y < eh / 2; ++y)
			for (int x = 0; x < ew / 2; ++x)
			{
				chroma[((size_t)y * (ew / 2) + x) * 2 + 0] = (uint8_t)(100 + (x * 70 / (ew / 2)));
				chroma[((size_t)y * (ew / 2) + x) * 2 + 1] = (uint8_t)(160 - (y * 70 / (eh / 2)));
			}

		reconstructor recon;
		recon.init_own(rl, rc);
		recon.init_cabac_own(cabac);

		// Reference reconstruction (slice-aware, same R).
		block_syntax bs;
		std::vector<uint8_t> recY((size_t)cw * ch), recCb((size_t)(cw / 2) * (ch / 2)), recCr((size_t)(cw / 2) * (ch / 2));
		recon.reconstruct(cfg, ew, eh, luma.data(), chroma.data(), bs, recY.data(), recCb.data(), recCr.data(), t.R);

		// Integrated GPU path.
		std::vector<std::vector<uint8_t>> payloads;
		recon.encode_frame(cfg, ew, eh, luma.data(), chroma.data(), t.R, payloads);

		std::vector<uint8_t> frame = build_parameter_sets(cfg);
		for (size_t s = 0; s < payloads.size(); ++s)
		{
			bitwriter hdr;
			write_slice_header(hdr, cfg, (uint32_t)(s * t.R) * nx, s == 0);
			std::vector<uint8_t> rbsp = hdr.bytes();
			rbsp.insert(rbsp.end(), payloads[s].begin(), payloads[s].end());
			std::vector<uint8_t> nal;
			emit_nal(nal, NAL_IDR_W_RADL, rbsp);
			frame.insert(frame.end(), nal.begin(), nal.end());
		}

		std::string base = "/tmp/ef_" + std::to_string(t.w) + "x" + std::to_string(t.h);
		std::string h265 = base + ".265", yuv = base + ".yuv";
		FILE * f = fopen(h265.c_str(), "wb");
		fwrite(frame.data(), 1, frame.size(), f);
		fclose(f);
		std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i " + h265 + " -f rawvideo -pix_fmt yuv420p " + yuv;
		int r = system(cmd.c_str());

		int mism = 0;
		bool ok = (r == 0);
		if (ok)
		{
			FILE * yf = fopen(yuv.c_str(), "rb");
			std::vector<uint8_t> dec((size_t)cw * ch * 3 / 2);
			size_t got = yf ? fread(dec.data(), 1, dec.size(), yf) : 0;
			if (yf) fclose(yf);
			if (got != dec.size()) ok = false;
			else
			{
				const uint8_t * dY = dec.data();
				const uint8_t * dCb = dY + (size_t)cw * ch;
				const uint8_t * dCr = dCb + (size_t)(cw / 2) * (ch / 2);
				for (size_t i = 0; i < (size_t)cw * ch; ++i) if (dY[i] != recY[i]) ++mism;
				for (size_t i = 0; i < (size_t)(cw / 2) * (ch / 2); ++i) if (dCb[i] != recCb[i]) ++mism;
				for (size_t i = 0; i < (size_t)(cw / 2) * (ch / 2); ++i) if (dCr[i] != recCr[i]) ++mism;
			}
		}
		bool pass = ok && mism == 0;
		if (not pass) ++fails;
		printf("%dx%d qp%d R=%d (%zu slices): frame=%zu bytes, decode %s, mismatch=%d  %s\n",
		       t.w, t.h, t.qp, t.R, payloads.size(), frame.size(), ok ? "ok" : "FAIL", mism, pass ? "PASS" : "FAIL");
	}
	printf("\n%s (%d failures)\n", fails == 0 ? "ENCODE_FRAME PIXEL-EXACT" : "FAILURES", fails);
	return fails == 0 ? 0 : 1;
}
