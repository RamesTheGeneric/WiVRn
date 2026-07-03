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

using namespace wivrn::h264;
using namespace wivrn::h267::gpu;

struct PC
{
	uint32_t mbw, mbh;
	int32_t qp, qpc;
	uint32_t cw, ch, ew, eh, diag, mbx_start;
};

int main(int argc, char ** argv)
{
	const char * spv = argc > 1 ? argv[1] : "/tmp/r264.spv";
	vk_compute vk;
	vk.init();
	auto pipe = vk.make_pipeline(spv, 11, sizeof(PC));

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
		auto bRY = vk.make_buffer((size_t)cw * ch * 4, true);
		auto bRCb = vk.make_buffer((size_t)cw2 * ch2 * 4, true);
		auto bRCr = vk.make_buffer((size_t)cw2 * ch2 * 4, true);
		auto bLDC = vk.make_buffer((size_t)nmb * 16 * 4, true);
		auto bLAC = vk.make_buffer((size_t)nmb * 256 * 4, true);
		auto bCDC = vk.make_buffer((size_t)nmb * 8 * 4, true);
		auto bCAC = vk.make_buffer((size_t)nmb * 128 * 4, true);
		auto bNL = vk.make_buffer((size_t)nmb * 16 * 4, true);
		auto bNC = vk.make_buffer((size_t)nmb * 8 * 4, true);

		// pack sources
		memcpy(bSY.ptr, img.Y.data(), (size_t)ew * eh);
		{
			uint8_t * c = (uint8_t *)bSC.ptr;
			for (int y = 0; y < ch2; ++y) for (int x = 0; x < cw2; ++x) {
				c[((size_t)y * cw2 + x) * 2 + 0] = img.Cb[(size_t)y * cw2 + x];
				c[((size_t)y * cw2 + x) * 2 + 1] = img.Cr[(size_t)y * cw2 + x];
			}
		}

		std::vector<vk_compute::buffer *> binds = {&bSY, &bSC, &bRY, &bRCb, &bRCr, &bLDC, &bLAC, &bCDC, &bCAC, &bNL, &bNC};
		std::vector<vk_compute::step> steps;
		const int qpc = wivrn::h264::xform::chroma_qp(t.qp);
		for (int d = 0; d <= mbw + mbh - 2; ++d) {
			int s = std::max(0, d - (mbh - 1)), e = std::min(d, mbw - 1);
			PC pc{(uint32_t)mbw, (uint32_t)mbh, t.qp, qpc, (uint32_t)cw, (uint32_t)ch, (uint32_t)ew, (uint32_t)eh, (uint32_t)d, (uint32_t)s};
			vk_compute::step st; st.gx = (uint32_t)(e - s + 1); st.gy = 1; st.gz = 1;
			st.push.resize(sizeof(pc)); memcpy(st.push.data(), &pc, sizeof(pc));
			steps.push_back(std::move(st));
		}
		vk.run_wavefront(pipe, binds, steps);

		int mism = 0;
		auto * gY = (int32_t *)bRY.ptr; auto * gCb = (int32_t *)bRCb.ptr; auto * gCr = (int32_t *)bRCr.ptr;
		for (size_t i = 0; i < (size_t)cw * ch; ++i) if ((uint8_t)gY[i] != cRecY[i]) ++mism;
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if ((uint8_t)gCb[i] != cRecCb[i]) ++mism;
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if ((uint8_t)gCr[i] != cRecCr[i]) ++mism;
		if (mism) ++fails;
		printf("%dx%d qp%d (%d MBs): recon mismatch=%d  %s\n", t.w, t.h, t.qp, nmb, mism, mism ? "FAIL" : "PASS");

		for (auto * b : binds) vk.destroy_buffer(*b);
	}
	printf("\n%s (%d failures)\n", fails == 0 ? "GPU RECON BIT-EXACT vs CPU" : "FAILURES", fails);
	return fails ? 1 : 0;
}
