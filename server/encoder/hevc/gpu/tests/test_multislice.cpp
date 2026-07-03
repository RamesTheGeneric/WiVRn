// End-to-end multi-slice validation: slice-aware GPU reconstruction + GPU CABAC
// (N slices) + slice assembly -> ffmpeg decode must equal the GPU reconstruction
// pixel-exact. That proves the decoder's independent-slice reconstruction matches
// ours (slice boundaries handled consistently in recon and entropy coding).
//
// Build/run (from server/encoder/hevc/):
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/hevc_recon_dc_luma.comp -o /tmp/rl.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/hevc_recon_dc_chroma.comp -o /tmp/rc.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/hevc_cabac.comp -o /tmp/cabac.spv
//   g++ -std=c++20 -I. gpu/tests/test_multislice.cpp gpu/gpu_reconstruct.cpp \
//       cabac_pass.cpp residual_coding.cpp cabac_reference.cpp param_sets.cpp bitwriter.cpp \
//       -lvulkan -o /tmp/test_multislice
//   /tmp/test_multislice /tmp/rl.spv /tmp/rc.spv /tmp/cabac.spv
#include "../../bitwriter.h"
#include "../../cabac_pass.h"
#include "../../param_sets.h"
#include "../gpu_reconstruct.h"
#include "../vk_compute.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace wivrn::h267;
using namespace wivrn::h267::gpu;

struct CPC
{
	int32_t qp;
	uint32_t bw, bh, ctbs_x, ctbs_y, slice_ctb_rows, out_stride;
};

int main(int argc, char ** argv)
{
	const char * rl = argv[1], *rc = argv[2], *cabac = argv[3];

	vk_compute vk;
	vk.init();
	auto pipe = vk.make_pipeline(cabac, 9, sizeof(CPC));

	int total_fail = 0;
	struct tc { int w, h, qp; };
	for (auto t: {tc{256, 256, 26}, tc{320, 256, 20}, tc{384, 320, 32}})
	{
		hevc_config cfg;
		cfg.width = t.w;
		cfg.height = t.h;
		cfg.qp = t.qp;
		cfg.bit_depth = 8;
		cfg.max_tb_log2_size = 3;
		const int cw = cfg.coded_width(), ch = cfg.coded_height();
		const int ew = t.w, eh = t.h; // display == a CTB multiple here, no padding
		const int bw = cw / 8, bh = ch / 8, nb = bw * bh;
		const uint32_t nx = cfg.ctbs_x(), ny = cfg.ctbs_y();

		// Synthetic source: gradients + texture so residuals are non-trivial.
		std::vector<uint8_t> luma((size_t)ew * eh);
		std::vector<uint8_t> chroma((size_t)ew * eh / 2);
		for (int y = 0; y < eh; ++y)
			for (int x = 0; x < ew; ++x)
			{
				int v = 40 + (x * 160 / ew) + ((x ^ y) & 31) + ((y / 13) * 7 % 40);
				luma[(size_t)y * ew + x] = (uint8_t)std::min(255, std::max(0, v));
			}
		for (int y = 0; y < eh / 2; ++y)
			for (int x = 0; x < ew / 2; ++x)
			{
				chroma[((size_t)y * (ew / 2) + x) * 2 + 0] = (uint8_t)(110 + (x * 60 / (ew / 2)));
				chroma[((size_t)y * (ew / 2) + x) * 2 + 1] = (uint8_t)(150 - (y * 60 / (eh / 2)));
			}

		reconstructor recon;
		recon.init_own(rl, rc);

		for (uint32_t R: {ny, 2u, 1u})
		{
			const uint32_t nsl = (ny + R - 1) / R;

			block_syntax bs;
			std::vector<uint8_t> recY((size_t)cw * ch), recCb((size_t)(cw / 2) * (ch / 2)), recCr((size_t)(cw / 2) * (ch / 2));
			recon.reconstruct(cfg, ew, eh, luma.data(), chroma.data(), bs,
			                  recY.data(), recCb.data(), recCr.data(), (int)R);

			// GPU CABAC, N slices.
			const uint32_t stride = (uint32_t)nb * 64u * 2u;
			auto bCbfL = vk.make_buffer((size_t)nb * 4);
			auto bCbfCb = vk.make_buffer((size_t)nb * 4);
			auto bCbfCr = vk.make_buffer((size_t)nb * 4);
			auto bLevY = vk.make_buffer((size_t)nb * 64 * 4);
			auto bLevCb = vk.make_buffer((size_t)nb * 16 * 4);
			auto bLevCr = vk.make_buffer((size_t)nb * 16 * 4);
			auto bOut = vk.make_buffer((size_t)stride * nsl * 4, true);
			auto bLen = vk.make_buffer((size_t)nsl * 4, true);
			auto bAvail = vk.make_buffer((size_t)nb * nsl * 4);
			for (int i = 0; i < nb; ++i)
			{
				((uint32_t *)bCbfL.ptr)[i] = bs.cbf_luma[i];
				((uint32_t *)bCbfCb.ptr)[i] = bs.cbf_cb[i];
				((uint32_t *)bCbfCr.ptr)[i] = bs.cbf_cr[i];
			}
			memcpy(bLevY.ptr, bs.lev_y.data(), (size_t)nb * 64 * 4);
			memcpy(bLevCb.ptr, bs.lev_cb.data(), (size_t)nb * 16 * 4);
			memcpy(bLevCr.ptr, bs.lev_cr.data(), (size_t)nb * 16 * 4);

			CPC pc{cfg.qp, (uint32_t)bw, (uint32_t)bh, nx, ny, R, stride};
			vk.run(pipe, {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail},
			       nsl, 1, 1, &pc, sizeof(pc));

			// Assemble the frame: parameter sets + one NAL per slice.
			std::vector<uint8_t> frame = build_parameter_sets(cfg);
			for (uint32_t s = 0; s < nsl; ++s)
			{
				bitwriter hdr;
				write_slice_header(hdr, cfg, s * R * nx, s == 0);
				std::vector<uint8_t> rbsp = hdr.bytes();
				uint32_t glen = ((uint32_t *)bLen.ptr)[s];
				for (uint32_t i = 0; i < glen; ++i)
					rbsp.push_back((uint8_t)(((uint32_t *)bOut.ptr)[s * stride + i] & 0xff));
				std::vector<uint8_t> nal;
				emit_nal(nal, NAL_IDR_W_RADL, rbsp);
				frame.insert(frame.end(), nal.begin(), nal.end());
			}

			// Write, decode with ffmpeg, compare to GPU reconstruction.
			std::string base = "/tmp/ms_" + std::to_string(t.w) + "x" + std::to_string(t.h) + "_R" + std::to_string(R);
			std::string h265 = base + ".265", yuv = base + ".yuv";
			FILE * f = fopen(h265.c_str(), "wb");
			fwrite(frame.data(), 1, frame.size(), f);
			fclose(f);
			std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i " + h265 +
			                  " -f rawvideo -pix_fmt yuv420p " + yuv + " 2>/dev/null";
			int rc2 = system(cmd.c_str());

			bool ok = (rc2 == 0);
			int mism = 0;
			if (ok)
			{
				FILE * yf = fopen(yuv.c_str(), "rb");
				std::vector<uint8_t> dec((size_t)cw * ch * 3 / 2);
				size_t got = yf ? fread(dec.data(), 1, dec.size(), yf) : 0;
				if (yf) fclose(yf);
				if (got != dec.size()) { ok = false; }
				else
				{
					// decoded planes at coded size (display==coded here)
					const uint8_t * dY = dec.data();
					const uint8_t * dCb = dY + (size_t)cw * ch;
					const uint8_t * dCr = dCb + (size_t)(cw / 2) * (ch / 2);
					for (size_t i = 0; i < (size_t)cw * ch; ++i) if (dY[i] != recY[i]) ++mism;
					for (size_t i = 0; i < (size_t)(cw / 2) * (ch / 2); ++i) if (dCb[i] != recCb[i]) ++mism;
					for (size_t i = 0; i < (size_t)(cw / 2) * (ch / 2); ++i) if (dCr[i] != recCr[i]) ++mism;
				}
			}
			bool pass = ok && mism == 0;
			if (not pass) ++total_fail;
			printf("%dx%d qp%d R=%u (%u slices): frame=%zu bytes, decode %s, mismatch=%d  %s\n",
			       t.w, t.h, t.qp, R, nsl, frame.size(), ok ? "ok" : "FAIL", mism, pass ? "PASS" : "FAIL");

			for (auto * b: {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail})
				vk.destroy_buffer(*b);
		}
	}
	printf("\n%s (%d failures)\n", total_fail == 0 ? "ALL DECODE PIXEL-EXACT" : "FAILURES", total_fail);
	return total_fail == 0 ? 0 : 1;
}
