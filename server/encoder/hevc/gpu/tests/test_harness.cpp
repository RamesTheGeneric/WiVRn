/*
 * Smoke test for the Vulkan compute harness: runs the trivial "double" shader
 * on the GPU and verifies out[i] == in[i]*2 + bias.
 * Build: glslangValidator -V ../shaders/double.comp -o double.spv
 *        g++ -std=c++20 -I.. test_harness.cpp -lvulkan -o test_harness
 */

#include "../vk_compute.h"

#include <cstdio>
#include <vector>

using namespace wivrn::h267::gpu;

int main(int argc, char ** argv)
{
	const char * spv = argc > 1 ? argv[1] : "double.spv";
	vk_compute vk;
	vk.init();

	const uint32_t N = 4096;
	auto in = vk.make_buffer(N * sizeof(int32_t));
	auto out = vk.make_buffer(N * sizeof(int32_t));
	auto * ip = (int32_t *)in.ptr;
	for (uint32_t i = 0; i < N; ++i)
		ip[i] = (int)i - 100;

	auto pipe = vk.make_pipeline(spv, 2, 8);
	struct
	{
		uint32_t count;
		int32_t bias;
	} pc{N, 7};
	vk.run(pipe, {&in, &out}, (N + 63) / 64, 1, 1, &pc, sizeof(pc));

	auto * op = (int32_t *)out.ptr;
	int fails = 0;
	for (uint32_t i = 0; i < N; ++i)
	{
		int expect = ((int)i - 100) * 2 + 7;
		if (op[i] != expect)
		{
			if (fails < 5)
				std::printf("FAIL i=%u got %d want %d\n", i, op[i], expect);
			++fails;
		}
	}
	std::printf("%s (%d/%u mismatches)\n", fails == 0 ? "HARNESS OK" : "HARNESS FAILED", fails, N);
	return fails == 0 ? 0 : 1;
}
