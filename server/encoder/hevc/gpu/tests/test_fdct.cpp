/*
 * Validate the GPU forward-DCT8+quant shader against the CPU reference
 * (transform.cpp) bit-for-bit over many random residual blocks and QPs.
 * Build: glslangValidator -V ../shaders/hevc_fdct8.comp -o fdct8.spv
 *        g++ -std=c++20 -I.. -I../.. test_fdct.cpp ../../transform.cpp -lvulkan -o test_fdct
 */

#include "../../transform.h"
#include "../vk_compute.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace wivrn::h267::gpu;

int main(int argc, char ** argv)
{
	const char * spv = argc > 1 ? argv[1] : "fdct8.spv";
	vk_compute vk;
	vk.init();

	const uint32_t nblocks = 8192;
	const int bd = 8;
	auto resb = vk.make_buffer(nblocks * 64 * sizeof(int32_t));
	auto levb = vk.make_buffer(nblocks * 64 * sizeof(int32_t));
	auto pipe = vk.make_pipeline(spv, 2, 12);

	auto * rp = (int32_t *)resb.ptr;
	// Deterministic pseudo-random residuals in [-255,255] with some structure.
	uint32_t seed = 12345;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
	for (uint32_t i = 0; i < nblocks * 64; ++i)
		rp[i] = (int)(rnd() % 511u) - 255;

	int total_fail = 0;
	for (int qp: {4, 12, 22, 26, 32, 40, 51})
	{
		struct
		{
			uint32_t nb;
			int32_t qp, bd;
		} pc{nblocks, qp, bd};
		vk.run(pipe, {&resb, &levb}, nblocks, 1, 1, &pc, sizeof(pc));

		auto * gpu = (int32_t *)levb.ptr;
		int fails = 0;
		for (uint32_t b = 0; b < nblocks; ++b)
		{
			int32_t coeff[64], cpu[64];
			wivrn::h267::xform::fdct(rp + b * 64, coeff, 8, bd);
			wivrn::h267::xform::quant(coeff, cpu, 8, qp, bd);
			for (int i = 0; i < 64; ++i)
				if (cpu[i] != gpu[b * 64 + i])
				{
					if (fails < 3)
						std::printf("  qp%d blk%u pos%d: cpu=%d gpu=%d\n", qp, b, i, cpu[i], gpu[b * 64 + i]);
					++fails;
				}
		}
		std::printf("qp%2d: %s (%d mismatches / %u coeffs)\n", qp, fails == 0 ? "MATCH" : "FAIL", fails, nblocks * 64);
		total_fail += fails;
	}
	std::printf("\n%s\n", total_fail == 0 ? "GPU FDCT8+QUANT BIT-EXACT vs CPU" : "FAILED");
	return total_fail == 0 ? 0 : 1;
}
