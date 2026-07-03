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

#include "video_encoder_hevc.h"

#include "encoder_settings.h"
#include "hevc/cpu_encoder.h"
#include "util/u_logging.h"
#include "utils/wivrn_vk_bundle.h"

#include <format>
#include <memory>
#include <span>
#include <stdexcept>

namespace wivrn
{

namespace
{
vk::raii::CommandPool make_cmd_pool(wivrn::vk_bundle & vk, uint8_t stream_idx)
{
	auto res = vk.device.createCommandPool(vk::CommandPoolCreateInfo{
	        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient,
	        .queueFamilyIndex = vk.transfer_queue ? vk.transfer_queue.family_index : vk.queue.family_index,
	});
	vk.name(res, std::format("hevc encoder {} command pool", stream_idx));
	return res;
}

int parse_qp(const encoder_settings & settings)
{
	auto it = settings.options.find("qp");
	if (it != settings.options.end())
	{
		try
		{
			int q = std::stoi(it->second);
			if (q >= 0 && q <= 51)
				return q;
		}
		catch (...)
		{
		}
	}
	return 26;
}
} // namespace

video_encoder_hevc::video_encoder_hevc(
        wivrn::vk_bundle & vk,
        const encoder_settings & settings,
        uint8_t stream_idx) :
        video_encoder(vk,
                      stream_idx,
                      vk.transfer_queue ? vk.transfer_queue.family_index : vk.queue.family_index,
                      settings,
                      std::make_unique<default_idr_handler>(),
                      false),
        vk{vk},
        cmd_pool{make_cmd_pool(vk, stream_idx)}
{
	if (settings.bit_depth != 8)
		throw std::runtime_error("hevc compute encoder only supports 8-bit encoding");
	if (settings.codec != h265)
		U_LOG_W("requested hevc encoder with codec != h265");

	luma_stride = extent.width;
	chroma_stride = extent.width;

	cfg.width = extent.width;
	cfg.height = extent.height;
	cfg.bit_depth = 8;
	cfg.qp = parse_qp(settings);
	cfg.max_tb_log2_size = 3; // this encoder uses 8x8 luma / 4x4 chroma TUs
	parameter_sets = hevc::build_parameter_sets(cfg);

	auto command_buffers = vk.device.allocateCommandBuffers(
	        {.commandPool = *cmd_pool, .commandBufferCount = num_slots});
	for (size_t i = 0; i < num_slots; ++i)
	{
		in[i].cmd = std::move(command_buffers[i]);
		vk.name(in[i].cmd, std::format("hevc {} transfer command buffer {}", stream_idx, i));
		in[i].fence = vk::raii::Fence(vk.device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		in[i].luma = buffer_allocation(
		        vk.device,
		        {.size = vk::DeviceSize(extent.width * extent.height), .usage = vk::BufferUsageFlagBits::eTransferDst},
		        {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST},
		        "hevc luma buffer");
		in[i].chroma = buffer_allocation(
		        vk.device,
		        {.size = vk::DeviceSize(extent.width * extent.height / 2), .usage = vk::BufferUsageFlagBits::eTransferDst},
		        {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST},
		        "hevc chroma buffer");
	}
}

void video_encoder_hevc::present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo compositor_sem, uint8_t slot, uint64_t)
{
	if (vk.device.waitForFences(*in[slot].fence, true, 1'000'000'000) == vk::Result::eTimeout)
	{
		U_LOG_E("Timeout on stream %d", stream_idx);
		return;
	}

	auto & cmd = in[slot].cmd;
	cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	if (need_transfer)
	{
		vk::ImageMemoryBarrier2 barrier{
		        .dstStageMask = vk::PipelineStageFlagBits2KHR::eTransfer,
		        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
		        .srcQueueFamilyIndex = vk.queue.family_index,
		        .dstQueueFamilyIndex = target_queue,
		        .image = y_cbcr,
		        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = stream_idx, .layerCount = 1},
		};
		cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
	}

	cmd.copyImageToBuffer(
	        y_cbcr, vk::ImageLayout::eGeneral, in[slot].luma,
	        vk::BufferImageCopy{
	                .bufferRowLength = luma_stride,
	                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::ePlane0, .baseArrayLayer = stream_idx, .layerCount = 1},
	                .imageExtent = {extent.width, extent.height, 1}});
	cmd.copyImageToBuffer(
	        y_cbcr, vk::ImageLayout::eGeneral, in[slot].chroma,
	        vk::BufferImageCopy{
	                .bufferRowLength = chroma_stride / 2,
	                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::ePlane1, .baseArrayLayer = stream_idx, .layerCount = 1},
	                .imageExtent = {extent.width / 2, extent.height / 2, 1}});
	cmd.end();

	std::unique_lock lock(vk.transfer_queue ? vk.transfer_queue.mutex : vk.queue.mutex);
	vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = *cmd};
	compositor_sem.stageMask = vk::PipelineStageFlagBits2::eTransfer;
	vk.device.resetFences(*in[slot].fence);
	(vk.transfer_queue ? vk.transfer_queue : vk.queue)
	        .queue.submit2(vk::SubmitInfo2{
	                               .waitSemaphoreInfoCount = 1,
	                               .pWaitSemaphoreInfos = &compositor_sem,
	                               .commandBufferInfoCount = 1,
	                               .pCommandBufferInfos = &cmd_info,
	                       },
	                       *in[slot].fence);
}

std::optional<video_encoder::data> video_encoder_hevc::encode(uint8_t slot, uint64_t frame_index)
{
	if (vk.device.waitForFences(*in[slot].fence, true, 1'000'000'000) == vk::Result::eTimeout)
	{
		U_LOG_E("Timeout on stream %d", stream_idx);
		return {};
	}

	const uint8_t * luma = reinterpret_cast<const uint8_t *>(in[slot].luma.map());
	const uint8_t * chroma = reinterpret_cast<const uint8_t *>(in[slot].chroma.map());

	// Assemble the CTB-aligned coded image, replicating edge samples into the
	// padding (which the conformance window crops out).
	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	const int ew = extent.width, eh = extent.height;
	hevc::yuv_image img;
	img.width = cw;
	img.height = ch;
	img.Y.resize((size_t)cw * ch);
	img.Cb.resize((size_t)(cw / 2) * (ch / 2));
	img.Cr.resize((size_t)(cw / 2) * (ch / 2));
	for (int y = 0; y < ch; ++y)
	{
		const int sy = y < eh ? y : eh - 1;
		for (int x = 0; x < cw; ++x)
		{
			const int sx = x < ew ? x : ew - 1;
			img.Y[(size_t)y * cw + x] = luma[(size_t)sy * luma_stride + sx];
		}
	}
	for (int y = 0; y < ch / 2; ++y)
	{
		const int sy = y < eh / 2 ? y : eh / 2 - 1;
		for (int x = 0; x < cw / 2; ++x)
		{
			const int sx = x < ew / 2 ? x : ew / 2 - 1;
			img.Cb[(size_t)y * (cw / 2) + x] = chroma[(size_t)sy * chroma_stride + sx * 2 + 0];
			img.Cr[(size_t)y * (cw / 2) + x] = chroma[(size_t)sy * chroma_stride + sx * 2 + 1];
		}
	}

	auto slice = hevc::encode_intra_frame(cfg, img);

	auto frame = std::make_shared<std::vector<uint8_t>>();
	frame->reserve(parameter_sets.size() + slice.size());
	frame->insert(frame->end(), parameter_sets.begin(), parameter_sets.end()); // VPS/SPS/PPS every IDR
	frame->insert(frame->end(), slice.begin(), slice.end());

	if (video_dump.is_open())
		video_dump.write(reinterpret_cast<const char *>(frame->data()), frame->size());

	(void)frame_index;
	return data{
	        .encoder = this,
	        .span = std::span<uint8_t>(*frame),
	        .mem = frame,
	        .prefer_control = true, // all-IDR: send reliably
	};
}

} // namespace wivrn
