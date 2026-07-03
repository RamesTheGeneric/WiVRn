/*
 * WiVRn VR streaming
 * Copyright (C) 2026  WiVRn contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "../cabac_pass.h" // block_syntax
#include "../param_sets.h" // hevc_config
#include "vk_compute.h"

#include <cstdint>

namespace wivrn::h267::gpu
{

// Runs the DC reconstruction wavefront (luma + chroma) on the GPU, turning a
// coded-size YUV frame into per-CU block_syntax {levels, cbf, mode} for the
// entropy stage. Uses the h2-67 compute shaders. Works either on its own device
// (offline testing on the target GPU) or adopting WiVRn's shared vk_bundle
// device (in the server).
class reconstructor
{
	vk_compute vkc;
	vk_compute::pipeline pl{}, pc{};
	bool ready = false;

	int alloc_cw = 0, alloc_ch = 0;
	vk_compute::buffer sY{}, rY{}, lY{}, cY{};
	vk_compute::buffer sCb{}, rCb{}, lCb{}, cCb{};
	vk_compute::buffer sCr{}, rCr{}, lCr{}, cCr{};

	void ensure_buffers(int cw, int ch);

public:
	// Wall-clock breakdown of the last reconstruct() call, in microseconds.
	struct timings
	{
		double upload_us = 0;   // memcpy source planes into host-visible GPU buffers
		double luma_us = 0;     // luma wavefront submit + wait
		double chroma_us = 0;   // Cb + Cr wavefronts submit + wait
		double readback_us = 0; // copy levels/cbf/recon back out
	};
	timings last_timings{};

	// WiVRn: use an existing device and the embedded SPIR-V of the two shaders.
	void init_adopt(VkPhysicalDevice phys, VkDevice dev, VkQueue queue, uint32_t qfam,
	                const uint32_t * luma_spv, size_t luma_words,
	                const uint32_t * chroma_spv, size_t chroma_words);

	// Offline testing: create an own device and load the two shaders from files.
	void init_own(const char * luma_spv_path, const char * chroma_spv_path);

	// Fill bs (levels/cbf, all-DC modes) from the coded-size source planes
	// (row-major int, luma cw*ch, chroma (cw/2)*(ch/2)). recon planes, if
	// non-null, receive the GPU reconstruction (coded size, one byte per sample).
	void reconstruct(const hevc_config & cfg,
	                 const int32_t * srcY, const int32_t * srcCb, const int32_t * srcCr,
	                 block_syntax & bs,
	                 uint8_t * recY = nullptr, uint8_t * recCb = nullptr, uint8_t * recCr = nullptr);
};

} // namespace wivrn::h267::gpu
