// M2 validation: GPU H.264 reconstruction wavefront must be bit-exact with the
// CPU reference (cpu_encoder264) reconstruction.
//
// Build/run (from server/encoder/h264/):
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_recon_dc.comp -o /tmp/r264.spv
//   g++ -std=c++20 -I. -I.. -I../hevc/gpu gpu/tests/test_recon264.cpp cpu_encoder264.cpp \
//       cavlc.cpp transform264.cpp param_sets264.cpp ../hevc/bitwriter.cpp -lvulkan -o /tmp/test_recon264
//   /tmp/test_recon264 /tmp/r264.spv
#include "../../../hevc/gpu/vk_compute.h"
#include "../../cpu_encoder264.h"
#include "../../param_sets264.h"
#include "../../transform264.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace wivrn::avc;
using namespace wivrn::h267::gpu;

struct PC
{
	uint32_t mbw, mbh;
	int32_t qp, qpc;
	uint32_t cw, ch, ew, eh, nmb, full_recon;
};

int main(int argc, char ** argv)
{
	const char * spv = argc > 1 ? argv[1] : "/tmp/r264.spv";
	vk_compute vk;
	vk.init();
	auto pipe = vk.make_pipeline(spv, 15, sizeof(PC));

	int fails = 0;
	struct tc { int w, h, qp; };
	for (auto t : {tc{64, 48, 26}, tc{128, 96, 20}, tc{256, 160, 34}, tc{768, 832, 26}})
	{
		h264_config cfg;
		cfg.width = t.w; cfg.height = t.h; cfg.qp = t.qp;
		const int cw = cfg.coded_width(), ch = cfg.coded_height(), cw2 = cw / 2, ch2 = ch / 2;
		const int ew = t.w, eh = t.h; // 16-multiple test sizes
		const int mbw = cfg.mb_width(), mbh = cfg.mb_height();

		yuv_image img;
		img.width = cw; img.height = ch;
		img.Y.resize((size_t)cw * ch); img.Cb.resize((size_t)cw2 * ch2); img.Cr.resize((size_t)cw2 * ch2);
		for (int y = 0; y < ch; ++y) for (int x = 0; x < cw; ++x)
			img.Y[(size_t)y * cw + x] = (uint8_t)(20 + (x * 180 / cw) + ((x * 7 + y * 13) & 63) + (y / 9) * 2);
		for (int y = 0; y < ch2; ++y) for (int x = 0; x < cw2; ++x) {
			img.Cb[(size_t)y * cw2 + x] = (uint8_t)(80 + (x * 90 / cw2) + ((x ^ y) & 31));
			img.Cr[(size_t)y * cw2 + x] = (uint8_t)(180 - (y * 90 / ch2) + ((x * 3) & 15));
		}

		std::vector<uint8_t> cRecY, cRecCb, cRecCr;
		encode_idr_frame(cfg, img, &cRecY, &cRecCb, &cRecCr);

		// GPU buffers
		auto round4 = [](size_t n) { return (n + 3) & ~size_t(3); };
		const int nmb = mbw * mbh;
		auto bSY = vk.make_buffer(round4((size_t)ew * eh));
		auto bSC = vk.make_buffer(round4((size_t)ew * eh / 2));
		auto bRY = vk.make_buffer(round4((size_t)cw * ch), true);      // packed u8
		auto bRCb = vk.make_buffer(round4((size_t)cw2 * ch2), true);   // packed u8
		auto bRCr = vk.make_buffer(round4((size_t)cw2 * ch2), true);   // packed u8
		auto bLDC = vk.make_buffer((size_t)nmb * 16 * 2, true);  // int16 levels
		auto bLAC = vk.make_buffer((size_t)nmb * 256 * 2, true); // int16 levels
		auto bCDC = vk.make_buffer((size_t)nmb * 8 * 2, true);   // int16 levels
		auto bCAC = vk.make_buffer((size_t)nmb * 128 * 2, true); // int16 levels
		auto bNL = vk.make_buffer((size_t)(cw / 4) * (ch / 4) * 4, true);
		auto bNC = vk.make_buffer((size_t)2 * (cw / 8) * (ch / 8) * 4, true);

		// pack sources
		memcpy(bSY.ptr, img.Y.data(), (size_t)ew * eh);
		{
			uint8_t * c = (uint8_t *)bSC.ptr;
			for (int y = 0; y < ch2; ++y) for (int x = 0; x < cw2; ++x) {
				c[((size_t)y * cw2 + x) * 2 + 0] = img.Cb[(size_t)y * cw2 + x];
				c[((size_t)y * cw2 + x) * 2 + 1] = img.Cr[(size_t)y * cw2 + x];
			}
		}

		// scoreboard: anti-diagonal MB order + claim counter + done flags
		auto bOrd = vk.make_buffer((size_t)nmb * 4);
		auto bClaim = vk.make_buffer(4, true);
		auto bDone = vk.make_buffer((size_t)nmb * 4, true);
		{
			auto * ord = (uint32_t *)bOrd.ptr;
			uint32_t k = 0;
			for (int d = 0; d <= mbw + mbh - 2; ++d)
				for (int mbx = std::max(0, d - (mbh - 1)); mbx <= std::min(d, mbw - 1); ++mbx)
					ord[k++] = uint32_t((d - mbx) * mbw + mbx);
			memset(bClaim.ptr, 0, 4);
			memset(bDone.ptr, 0, (size_t)nmb * 4);
		}
		auto bHalo = vk.make_buffer((size_t)nmb * 16 * 4, true);
		std::vector<vk_compute::buffer *> binds = {&bSY, &bSC, &bRY, &bRCb, &bRCr, &bLDC, &bLAC, &bCDC, &bCAC, &bNL, &bNC, &bOrd, &bClaim, &bDone, &bHalo};
		const int qpc = wivrn::avc::xform::chroma_qp(t.qp);
		PC pc{(uint32_t)mbw, (uint32_t)mbh, t.qp, qpc, (uint32_t)cw, (uint32_t)ch, (uint32_t)ew, (uint32_t)eh, (uint32_t)nmb, 1u};
		vk.run(pipe, binds, std::min(nmb, 64), 1, 1, &pc, sizeof(pc));

		int mism = 0;
		auto * gY = (uint8_t *)bRY.ptr; auto * gCb = (uint8_t *)bRCb.ptr; auto * gCr = (uint8_t *)bRCr.ptr; // packed u8
		for (size_t i = 0; i < (size_t)cw * ch; ++i) if (gY[i] != cRecY[i]) ++mism;
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if (gCb[i] != cRecCb[i]) ++mism;
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if (gCr[i] != cRecCr[i]) ++mism;
		if (mism) ++fails;
		printf("%dx%d qp%d (%d MBs): recon mismatch=%d  %s\n", t.w, t.h, t.qp, nmb, mism, mism ? "FAIL" : "PASS");

		for (auto * b : binds) vk.destroy_buffer(*b);
	}
	printf("\n%s (%d failures)\n", fails == 0 ? "GPU RECON BIT-EXACT vs CPU" : "FAILURES", fails);
	return fails ? 1 : 0;
}
