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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace wivrn::h267;
using namespace wivrn::h267::gpu;

struct PC
{
	int32_t qp;
	uint32_t bw, bh, ctbs_x, ctbs_y, slice_ctb_rows, out_stride;
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

	int failures = 0, checks = 0;
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

			// Try several slice heights: whole frame, 2 rows/slice, 1 row/slice.
			for (uint32_t R: {ny, 2u, 1u})
			{
				const uint32_t nsl = (ny + R - 1) / R;
				const uint32_t stride = (uint32_t)nb * 64u * 2u; // generous byte cap per slice

				auto bCbfL = vk.make_buffer((size_t)nb * 4);
				auto bCbfCb = vk.make_buffer((size_t)nb * 4);
				auto bCbfCr = vk.make_buffer((size_t)nb * 4);
				auto bLevY = vk.make_buffer((size_t)nb * 64 * 4);
				auto bLevCb = vk.make_buffer((size_t)nb * 16 * 4);
				auto bLevCr = vk.make_buffer((size_t)nb * 16 * 4);
				auto bOut = vk.make_buffer((size_t)stride * nsl * 4, /*cached=*/true);
				auto bLen = vk.make_buffer((size_t)nsl * 4, /*cached=*/true);
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

				PC pc{cfg.qp, (uint32_t)bw, (uint32_t)bh, nx, ny, R, stride};
				vk.run(pipe, {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail},
				       nsl, 1, 1, &pc, sizeof(pc));

				bool all_ok = true;
				for (uint32_t s = 0; s < nsl; ++s)
				{
					const int row0 = (int)(s * R);
					const int rows = (int)std::min(R, ny - s * R);
					auto ref = encode_slice_payload(cfg, bs, row0, rows, s + 1 == nsl);
					uint32_t glen = ((uint32_t *)bLen.ptr)[s];
					bool ok = (glen == ref.size());
					if (ok)
						for (uint32_t i = 0; i < glen; ++i)
							if ((uint8_t)(((uint32_t *)bOut.ptr)[s * stride + i] & 0xff) != ref[i]) { ok = false; break; }
					if (not ok) all_ok = false;
					++checks;
				}
				if (not all_ok) ++failures;
				printf("%dx%d qp%d seed%u R=%u (%u slices): %s\n",
				       tc.w, tc.h, tc.qp, seed, R, nsl, all_ok ? "OK" : "MISMATCH");

				for (auto * b: {&bCbfL, &bCbfCb, &bCbfCr, &bLevY, &bLevCb, &bLevCr, &bOut, &bLen, &bAvail})
					vk.destroy_buffer(*b);
			}
		}
	}

	printf("\n%s (%d slice-checks, %d failures)\n", failures == 0 ? "ALL BYTE-EXACT" : "FAILURES", checks, failures);
	return failures == 0 ? 0 : 1;
}
