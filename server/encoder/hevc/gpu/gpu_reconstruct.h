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
	vk_compute::pipeline pl{}, pc{}, pcab{};
	bool ready = false, cabac_ready = false;

	int alloc_cw = 0, alloc_ch = 0;
	vk_compute::buffer sY{}, rY{}, lY{}, cY{}; // sY: raw uint8 luma (extent)
	vk_compute::buffer sChroma{};              // raw uint8 CbCr interleaved (extent)
	vk_compute::buffer rCb{}, lCb{}, cCb{};
	vk_compute::buffer rCr{}, lCr{}, cCr{};

	// GPU CABAC output buffers (per-slice byte streams + lengths + avail scratch).
	int cab_slices = 0;
	uint32_t cab_stride = 0;
	vk_compute::buffer bOut{}, bLen{}, bAvail{};

	void ensure_buffers(int cw, int ch, int ew, int eh);
	void ensure_cabac_buffers(int nslices, int nb, uint32_t stride);

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

	// Add the GPU CABAC pipeline (enables encode_frame). Call after init_*.
	void init_cabac_adopt(const uint32_t * cabac_spv, size_t cabac_words);
	void init_cabac_own(const char * cabac_spv_path);

	// Fill bs (levels/cbf, all-DC modes) directly from the raw compositor planes
	// at extent resolution ew x eh: lumaU8 is the 8-bit Y plane (ew*eh, stride ew),
	// chromaU8 is the 8-bit interleaved CbCr plane (ew*eh/2, i.e. (ew/2)*(eh/2)
	// pairs). The shaders edge-clamp-sample these up to the coded size, so no CPU
	// padding/de-interleaving is needed. recon planes, if non-null, receive the GPU
	// reconstruction (coded size, one byte per sample).
	// slice_ctb_rows > 0 partitions the picture into horizontal slices of that
	// many CTB rows (the last is shorter if it doesn't divide), restricting intra
	// prediction to within each slice so the reconstruction matches a decoder that
	// decodes the slices independently. <=0 means a single slice for the frame.
	void reconstruct(const hevc_config & cfg,
	                 int ew, int eh,
	                 const uint8_t * lumaU8, const uint8_t * chromaU8,
	                 block_syntax & bs,
	                 uint8_t * recY = nullptr, uint8_t * recCb = nullptr, uint8_t * recCr = nullptr,
	                 int slice_ctb_rows = 0);

	// Full GPU path: reconstruct then CABAC-encode entirely on the GPU (the level
	// buffers never leave the device), returning one CABAC payload per slice. The
	// caller prepends slice headers + NAL framing. Requires init_cabac_*.
	void encode_frame(const hevc_config & cfg,
	                  int ew, int eh,
	                  const uint8_t * lumaU8, const uint8_t * chromaU8,
	                  int slice_ctb_rows,
	                  std::vector<std::vector<uint8_t>> & slice_payloads);
};

} // namespace wivrn::h267::gpu
