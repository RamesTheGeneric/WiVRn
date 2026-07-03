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

#include "video_encoder_h2_67.h"

#include "encoder_settings.h"
#include "util/u_logging.h"
#include "utils/wivrn_vk_bundle.h"
#include "wivrn-server_shaders.h" // ::shaders (embedded SPIR-V)

#include <chrono>
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
	vk.name(res, std::format("h2-67 encoder {} command pool", stream_idx));
	return res;
}

int parse_qp(const encoder_settings & settings)
{
	if (auto it = settings.options.find("qp"); it != settings.options.end())
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

video_encoder_h2_67::video_encoder_h2_67(
        wivrn::vk_bundle & vk,
        const encoder_settings & settings,
        uint8_t stream_idx) :
        video_encoder(vk,
                      stream_idx,
                      vk.transfer_queue ? vk.transfer_queue.family_index : vk.queue.family_index,
                      settings,
                      std::make_unique<default_idr_handler>(),
                      true),
        vk{vk},
        cmd_pool{make_cmd_pool(vk, stream_idx)}
{
	if (settings.bit_depth != 8)
		throw std::runtime_error("h2-67 encoder only supports 8-bit encoding");
	if (settings.codec != h265)
		U_LOG_W("requested h2-67 encoder with codec != h265");

	luma_stride = extent.width;
	chroma_stride = extent.width;

	cfg.width = extent.width;
	cfg.height = extent.height;
	cfg.bit_depth = 8;
	cfg.qp = parse_qp(settings);
	cfg.max_tb_log2_size = 3;
	parameter_sets = h267::build_parameter_sets(cfg);

	// Reconstruction runs on WiVRn's shared device with the embedded shaders.
	const auto & luma_spv = ::shaders.at("hevc_recon_dc_luma");
	const auto & chroma_spv = ::shaders.at("hevc_recon_dc_chroma");
	recon.init_adopt(
	        static_cast<VkPhysicalDevice>(*vk.physical_device),
	        static_cast<VkDevice>(*vk.device),
	        static_cast<VkQueue>(*vk.queue.queue),
	        vk.queue.family_index,
	        luma_spv.data(), luma_spv.size(),
	        chroma_spv.data(), chroma_spv.size());

	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	src_y.resize((size_t)cw * ch);
	src_cb.resize((size_t)(cw / 2) * (ch / 2));
	src_cr.resize((size_t)(cw / 2) * (ch / 2));

	auto command_buffers = vk.device.allocateCommandBuffers(
	        {.commandPool = *cmd_pool, .commandBufferCount = num_slots});
	for (size_t i = 0; i < num_slots; ++i)
	{
		in[i].cmd = std::move(command_buffers[i]);
		vk.name(in[i].cmd, std::format("h2-67 {} transfer command buffer {}", stream_idx, i));
		in[i].fence = vk::raii::Fence(vk.device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		in[i].luma = buffer_allocation(
		        vk.device,
		        {.size = vk::DeviceSize(extent.width * extent.height), .usage = vk::BufferUsageFlagBits::eTransferDst},
		        {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST},
		        "h2-67 luma buffer");
		in[i].chroma = buffer_allocation(
		        vk.device,
		        {.size = vk::DeviceSize(extent.width * extent.height / 2), .usage = vk::BufferUsageFlagBits::eTransferDst},
		        {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST},
		        "h2-67 chroma buffer");
	}
}

void video_encoder_h2_67::present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo compositor_sem, uint8_t slot, uint64_t)
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

std::optional<video_encoder::data> video_encoder_h2_67::encode(uint8_t slot, uint64_t frame_index)
{
	if (vk.device.waitForFences(*in[slot].fence, true, 1'000'000'000) == vk::Result::eTimeout)
	{
		U_LOG_E("Timeout on stream %d", stream_idx);
		return {};
	}

	const uint8_t * luma = reinterpret_cast<const uint8_t *>(in[slot].luma.map());
	const uint8_t * chroma = reinterpret_cast<const uint8_t *>(in[slot].chroma.map());

	using clk = std::chrono::steady_clock;
	auto us = [](clk::time_point a, clk::time_point b) {
		return std::chrono::duration<double, std::micro>(b - a).count();
	};
	auto t_build0 = clk::now();

	// Build the CTB-aligned coded-size source planes (int), replicating edge
	// samples into the padding, which the conformance window crops out.
	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	const int ew = extent.width, eh = extent.height;
	for (int y = 0; y < ch; ++y)
	{
		const int sy = y < eh ? y : eh - 1;
		for (int x = 0; x < cw; ++x)
			src_y[(size_t)y * cw + x] = luma[(size_t)sy * luma_stride + (x < ew ? x : ew - 1)];
	}
	for (int y = 0; y < ch / 2; ++y)
	{
		const int sy = y < eh / 2 ? y : eh / 2 - 1;
		for (int x = 0; x < cw / 2; ++x)
		{
			const int sx = x < ew / 2 ? x : ew / 2 - 1;
			src_cb[(size_t)y * (cw / 2) + x] = chroma[(size_t)sy * chroma_stride + sx * 2 + 0];
			src_cr[(size_t)y * (cw / 2) + x] = chroma[(size_t)sy * chroma_stride + sx * 2 + 1];
		}
	}

	auto t_recon0 = clk::now();
	// GPU reconstruction wavefront -> per-CU levels/cbf. The shared queue is
	// mutex-protected; the reconstructor submits and waits on it.
	{
		std::unique_lock lock(vk.queue.mutex);
		recon.reconstruct(cfg, src_y.data(), src_cb.data(), src_cr.data(), bs);
	}

	auto t_cabac0 = clk::now();
	// CPU entropy coding from the level buffers.
	auto slice = h267::encode_slice_from_syntax(cfg, bs);
	auto t_cabac1 = clk::now();

	auto frame = std::make_shared<std::vector<uint8_t>>();
	frame->reserve(parameter_sets.size() + slice.size());
	frame->insert(frame->end(), parameter_sets.begin(), parameter_sets.end());
	frame->insert(frame->end(), slice.begin(), slice.end());

	// Note: the base class writes the returned span to WIVRN_DUMP_VIDEO in SendData().

	// Rolling per-stage profile.
	const auto & rt = recon.last_timings;
	prof.build += us(t_build0, t_recon0);
	prof.recon += us(t_recon0, t_cabac0);
	prof.cabac += us(t_cabac0, t_cabac1);
	prof.up += rt.upload_us;
	prof.luma += rt.luma_us;
	prof.chroma += rt.chroma_us;
	prof.rb += rt.readback_us;
	prof.bytes += (double)slice.size();
	// Opt-in per-stage profiling: set WIVRN_H267_PROFILE=1 to log stage averages.
	static const bool profile_enabled = std::getenv("WIVRN_H267_PROFILE") != nullptr;
	if (profile_enabled && ++prof.n >= prof_window)
	{
		const double n = prof.n;
		U_LOG_W("h2-67[%u] %.0fx%u avg/frame: build=%.0fus recon=%.0fus (up=%.0f luma=%.0f chroma=%.0f rb=%.0f) cabac=%.0fus | total=%.2fms slice=%.0fKB",
		        stream_idx, (double)extent.width, extent.height,
		        prof.build / n, prof.recon / n, prof.up / n, prof.luma / n, prof.chroma / n, prof.rb / n,
		        prof.cabac / n, (prof.build + prof.recon + prof.cabac) / n / 1000.0, prof.bytes / n / 1024.0);
		prof = {};
	}

	(void)frame_index;
	return data{
	        .encoder = this,
	        .span = std::span<uint8_t>(*frame),
	        .mem = frame,
	        .prefer_control = true,
	};
}

} // namespace wivrn
