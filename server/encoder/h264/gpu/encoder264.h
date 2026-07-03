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

#include "../../hevc/gpu/vk_compute.h"
#include "../param_sets264.h"

#include <cstdint>
#include <vector>

namespace wivrn::avc::gpu
{

// Full GPU H.264 encoder: reconstruction wavefront + parallel CAVLC (emit +
// prefix-sum + stitch) entirely on the GPU. The CPU only builds the tiny slice
// header, appends the stitched payload, and does emulation prevention + NAL
// framing. Uses the same vk_compute harness as the HEVC path.
class encoder
{
	using vk_compute = wivrn::h267::gpu::vk_compute;
	vk_compute vkc;
	vk_compute::pipeline recon{}, emit{}, prefix{}, stitch{};
	bool ready = false;

	int alloc_cw = 0, alloc_ch = 0, alloc_ew = 0;
	// source (write-combined) + everything the CPU reads back (cached)
	vk_compute::buffer sY{}, sChroma{};
	vk_compute::buffer rY{}, rCb{}, rCr{};
	vk_compute::buffer lDC{}, lAC{}, cDC{}, cAC{}, nnzL{}, nnzC{};
	vk_compute::buffer scratch{}, bitLen{}, offset{}, total{}, outbits{};
	// Scoreboard reconstruction: anti-diagonal MB order + claim counter + done flags.
	vk_compute::buffer mbOrder{}, claim{}, doneBuf{};
	uint32_t stride_words = 256;

	void ensure_buffers(const h264_config & cfg, int ew, int eh);

public:
	// Adopt WiVRn's device + the four embedded shaders' SPIR-V.
	void init_adopt(VkPhysicalDevice phys, VkDevice dev, VkQueue queue, uint32_t qfam,
	                const uint32_t * recon_spv, size_t recon_words,
	                const uint32_t * emit_spv, size_t emit_words,
	                const uint32_t * prefix_spv, size_t prefix_words,
	                const uint32_t * stitch_spv, size_t stitch_words);

	// Offline testing: own device + shader file paths.
	void init_own(const char * recon_path, const char * emit_path, const char * prefix_path, const char * stitch_path);

	// Encode one all-IDR frame from the raw compositor planes (8-bit Y at extent
	// resolution ew x eh, interleaved CbCr) and return the complete Annex-B frame
	// (SPS + PPS + IDR slice). Optionally returns the reconstruction.
	std::vector<uint8_t> encode_frame(const h264_config & cfg, int ew, int eh,
	                                  const uint8_t * lumaU8, const uint8_t * chromaU8,
	                                  uint8_t * recY = nullptr, uint8_t * recCb = nullptr, uint8_t * recCr = nullptr);

	// Timing (microseconds) of the last encode_frame's stages.
	struct timings { double upload = 0, recon_us = 0, cavlc_us = 0, assemble_us = 0;
	                 double emit_us = 0, prefix_us = 0, stitch_us = 0; } last_timings{};
};

} // namespace wivrn::avc::gpu
