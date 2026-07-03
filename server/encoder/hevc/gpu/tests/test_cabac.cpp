// Byte-exact validation of the GPU CABAC shader (hevc_cabac.comp) against the
// CPU reference (encode_slice_payload). Builds a consistent synthetic
// block_syntax, codes it both ways, and asserts identical bytes.
//
// Build/run (from server/encoder/hevc/):
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/hevc_cabac.comp -o /tmp/cabac.spv
//   g++ -std=c++20 -I. gpu/tests/test_cabac.cpp cabac_pass.cpp residual_coding.cpp \
//       cabac_reference.cpp param_sets.cpp bitwriter.cpp -lvulkan -o /tmp/test_cabac
//   /tmp/test_cabac /tmp/cabac.spv
#include "../../cabac_pass.h"
#include "../../param_sets.h"
#include "../vk_compute.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace wivrn::h267;
using namespace wivrn::h267::gpu;

struct PC
{
	int32_t qp;
	uint32_t bw, bh, ctbs_x, ctb_row0, ctb_rows, out_stride, last_slice;
};

// Deterministic sparse-but-consistent block_syntax: random small coefficients,
// cbf set iff the block has any nonzero coeff (as the real encoder guarantees).
static block_syntax make_syntax(int nb, uint32_t seed)
{
	uint32_t s = seed;
	auto rnd = [&]() { s = s * 1664525u + 1013904223u; return s; };
	block_syntax bs;
	bs.mode.assign(nb, 1); // DC
	bs.cbf_luma.assign(nb, 0);
	bs.cbf_cb.assign(nb, 0);
	bs.cbf_cr.assign(nb, 0);
	bs.lev_y.assign((size_t)nb * 64, 0);
	bs.lev_cb.assign((size_t)nb * 16, 0);
	bs.lev_cr.assign((size_t)nb * 16, 0);
	auto fill = [&](int32_t * lev, int cnt, uint8_t & cbf) {
		int any = 0;
		for (int i = 0; i < cnt; ++i)
		{
			uint32_t r = rnd();
			if ((r & 7u) == 0u) // ~1/8 coeffs nonzero
			{
				int mag = 1 + (int)((r >> 4) % ((r % 40u == 0u) ? 40u : 3u)); // occasional large
				int v = ((r >> 3) & 1u) ? -mag : mag;
				lev[i] = v;
				any = 1;
			}
		}
		cbf = (uint8_t)any;
	};
	for (int b = 0; b < nb; ++b)
	{
		fill(&bs.lev_y[(size_t)b * 64], 64, bs.cbf_luma[b]);
		fill(&bs.lev_cb[(size_t)b * 16], 16, bs.cbf_cb[b]);
		fill(&bs.lev_cr[(size_t)b * 16], 16, bs.cbf_cr[b]);
	}
	return bs;
}

int main(int argc, char ** argv)
{
	const char * spv = argc > 1 ? argv[1] : "/tmp/cabac.spv";

	vk_compute vk;
	vk.init();
	auto pipe = vk.make_pipeline(spv, 9, sizeof(PC));

	struct testcase
	{
		int w, h, qp;
	};
	const testcase cases[] = {
	        {128, 128, 26}, {192, 128, 20}, {256, 192, 32}, {320, 256, 12}, {128, 320, 40}};

	int failures = 0;
	for (auto tc: cases)
	{
		for (uint32_t seed: {1u, 7u, 99u})
		{
			hevc_config cfg;
			cfg.width = tc.w;
			cfg.height = tc.h;
			cfg.qp = tc.qp;
			cfg.bit_depth = 8;
			cfg.max_tb_log2_size = 3;
			const int cw = cfg.coded_width(), ch = cfg.coded_height();
			const int bw = cw / 8, bh = ch / 8, nb = bw * bh;
			const uint32_t ny = cfg.ctbs_y(), nx = cfg.ctbs_x();

			block_syntax bs = make_syntax(nb, seed + tc.w * 131 + tc.h);

			// CPU reference: whole frame as one slice.
			auto ref = encode_slice_payload(cfg, bs, 0, (int)ny, true);

			// GPU: 1 workgroup covering all CTB rows.
			const uint32_t stride = (uint32_t)nb * 64u * 2u; // generous byte cap
			auto bCbfL = vk.make_buffer((size_t)nb * 4);
			auto bCbfCb = vk.make_buffer((size_t)nb * 4);
			auto bCbfCr = vk.make_buffer((size_t)nb * 4);
			auto bLevY = vk.make_buffer((size_t)nb * 64 * 4);
			auto bLevCb = vk.make_buffer((size_t)nb * 16 * 4);
			auto bLevCr = vk.make_buffer((size_t)nb * 16 * 4);
			auto bOut = vk.make_buffer((size_t)stride * 4, /*cached=*/true);
			auto bLen = vk.make_buffer(4, /*cached=*/true);
			auto bAvail = vk.make_buffer((size_t)nb * 4);

			for (int i = 0; i < nb; ++i)
			{
				((uint32_t *)bCbfL.ptr)[i] = bs.cbf_luma[i];
				((uint32_t *)bCbfCb.ptr)[i] = bs.cbf_cb[i];
				((uint32_t *)bCbfCr.ptr)[i] = bs.cbf_cr[i];
			}
			memcpy(bLevY.ptr, bs.lev_y.data(), (size_t)nb * 64 * 4);
			memcpy(bLevCb.ptr, bs.lev_cb.data(), (size_t)nb * 16 * 4);
			memcpy(bLevCr.ptr, bs.lev_cr.data(), (size_t)nb * 16 * 4);

			PC pc{cfg.qp, (uint32_t)bw, (uint32_t)bh, nx, 0u, ny, stride, 1u};
			vk.run(pipe, {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail},
			       1, 1, 1, &pc, sizeof(pc));

			uint32_t glen = ((uint32_t *)bLen.ptr)[0];
			std::vector<uint8_t> gpu(glen);
			for (uint32_t i = 0; i < glen; ++i)
				gpu[i] = (uint8_t)(((uint32_t *)bOut.ptr)[i] & 0xff);

			bool ok = (gpu.size() == ref.size());
			size_t firstdiff = ref.size();
			if (ok)
				for (size_t i = 0; i < ref.size(); ++i)
					if (gpu[i] != ref[i])
					{
						ok = false;
						firstdiff = i;
						break;
					}

			printf("%dx%d(coded %dx%d) qp%d seed%u: cpu=%zu gpu=%u  %s",
			       tc.w, tc.h, cw, ch, tc.qp, seed, ref.size(), glen, ok ? "OK\n" : "MISMATCH");
			if (not ok)
			{
				++failures;
				if (firstdiff < ref.size())
					printf("  first diff at byte %zu: cpu=%02x gpu=%02x\n",
					       firstdiff, ref[firstdiff], firstdiff < gpu.size() ? gpu[firstdiff] : 0);
				else
					printf("  size differs\n");
			}

			for (auto * b: {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail})
				vk.destroy_buffer(*b);
		}
	}

	printf("\n%s (%d failures)\n", failures == 0 ? "ALL BYTE-EXACT" : "FAILURES", failures);
	return failures == 0 ? 0 : 1;
}
