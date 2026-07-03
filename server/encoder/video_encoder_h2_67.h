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

#include "hevc/cabac_pass.h"
#include "hevc/gpu/gpu_reconstruct.h"
#include "hevc/param_sets.h"
#include "video_encoder.h"
#include "vk/allocation.h"

#include <array>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

namespace wivrn
{

// "h2-67": a from-scratch, shader-based HEVC intra encoder for machines that
// have GPU compute but no hardware video-encode block. The per-block
// reconstruction (intra prediction, transform, quantisation, inverse transform)
// runs as a Vulkan compute wavefront on WiVRn's shared device; entropy coding
// (CABAC) runs on the CPU from the reconstruction's level buffers. The output is
// standard Main-profile HEVC decodable by the existing client decoders.
class video_encoder_h2_67 : public video_encoder
{
	wivrn::vk_bundle & vk;
	vk::raii::CommandPool cmd_pool;

	struct in_t
	{
		vk::raii::Fence fence = nullptr;
		vk::raii::CommandBuffer cmd = nullptr;
		buffer_allocation luma;   // copied Y plane (8-bit, extent-sized)
		buffer_allocation chroma; // copied CbCr plane (interleaved, half res)
	};
	std::array<in_t, num_slots> in;

	uint32_t luma_stride;
	uint32_t chroma_stride;

	hevc::hevc_config cfg;
	std::vector<uint8_t> parameter_sets;

	hevc::gpu::reconstructor recon;
	// Reused host-side coded-size source planes and syntax across frames.
	std::vector<int32_t> src_y, src_cb, src_cr;
	hevc::block_syntax bs;

public:
	video_encoder_h2_67(wivrn::vk_bundle & vk, const encoder_settings & settings, uint8_t stream_idx);

	void present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo, uint8_t slot, uint64_t frame_index) override;

	std::optional<data> encode(uint8_t slot, uint64_t frame_index) override;
};

} // namespace wivrn
