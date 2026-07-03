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

#include "hevc/param_sets.h"
#include "video_encoder.h"
#include "vk/allocation.h"

#include <array>
#include <vulkan/vulkan_raii.hpp>

namespace wivrn
{

// From-scratch HEVC intra encoder (Main profile, 8-bit, all-IDR). The compressed
// bitstream is produced with our own conformance-verified encoder rather than a
// hardware video-encode block, for machines that have GPU compute but no
// hardware encoder. This build copies the compositor's YUV image to the host and
// runs the CPU reference encoder; the reconstruction stage is being moved to
// Vulkan compute (server/encoder/hevc/gpu). Bitstream is standard Main-profile
// HEVC decodable by the existing client decoders.
class video_encoder_hevc : public video_encoder
{
	wivrn::vk_bundle & vk;
	uint64_t sem_value = 0;
	vk::raii::CommandPool cmd_pool;

	struct in_t
	{
		vk::raii::Fence fence = nullptr;
		vk::raii::CommandBuffer cmd = nullptr;
		buffer_allocation luma;   // copied Y plane (8-bit, extent-sized)
		buffer_allocation chroma; // copied CbCr plane (interleaved, half res)
	};
	std::array<in_t, num_slots> in;

	uint32_t luma_stride;   // = extent.width
	uint32_t chroma_stride; // bytes per chroma row = extent.width

	hevc::hevc_config cfg;
	std::vector<uint8_t> parameter_sets; // built once (config is static)

public:
	video_encoder_hevc(wivrn::vk_bundle & vk, const encoder_settings & settings, uint8_t stream_idx);

	void present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo, uint8_t slot, uint64_t frame_index) override;

	std::optional<data> encode(uint8_t slot, uint64_t frame_index) override;
};

} // namespace wivrn
