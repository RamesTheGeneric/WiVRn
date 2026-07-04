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
	use_h264 = (settings.codec == h264);

	luma_stride = extent.width;
	chroma_stride = extent.width;

	const auto vphys = static_cast<VkPhysicalDevice>(*vk.physical_device);
	const auto vdev = static_cast<VkDevice>(*vk.device);
	// A dedicated compute queue per eye if available, else the shared main queue.
	auto & enc_q = (stream_idx < vk.compute_queues.size()) ? vk.compute_queues[stream_idx] : vk.queue;
	enc_queue_mutex = &enc_q.mutex;
	const auto vqueue = static_cast<VkQueue>(*enc_q.queue);
	const uint32_t vqfam = enc_q.family_index;

	if (use_h264)
	{
		h264_cfg.width = extent.width;
		h264_cfg.height = extent.height;
		h264_cfg.bit_depth = 8;
		h264_cfg.qp = parse_qp(settings);
		const auto & r = ::shaders.at("h264_recon_dc");
		const auto & e = ::shaders.at("h264_cavlc_emit");
		const auto & p = ::shaders.at("h264_cavlc_prefix");
		const auto & s = ::shaders.at("h264_cavlc_stitch");
		h264_enc.init_adopt(vphys, vdev, vqueue, vqfam,
		                    r.data(), r.size(), e.data(), e.size(), p.data(), p.size(), s.data(), s.size());
	}
	else
	{
		cfg.width = extent.width;
		cfg.height = extent.height;
		cfg.bit_depth = 8;
		cfg.qp = parse_qp(settings);
		cfg.max_tb_log2_size = 3;
		parameter_sets = h267::build_parameter_sets(cfg);

		// Reconstruction + CABAC run on WiVRn's shared device with the embedded shaders.
		const auto & luma_spv = ::shaders.at("hevc_recon_dc_luma");
		const auto & chroma_spv = ::shaders.at("hevc_recon_dc_chroma");
		const auto & cabac_spv = ::shaders.at("hevc_cabac");
		recon.init_adopt(vphys, vdev, vqueue, vqfam,
		                 luma_spv.data(), luma_spv.size(),
		                 chroma_spv.data(), chroma_spv.size());
		recon.init_cabac_adopt(cabac_spv.data(), cabac_spv.size());
	}

	// CTB rows per independent slice: fewer = better compression, more = more GPU
	// CABAC parallelism. 1 row/slice maximises parallelism (CABAC is serial within
	// a slice, so slice count is the only parallelism). Configurable via the
	// "slice-rows" encoder option.
	slice_ctb_rows = 1;
	if (auto it = settings.options.find("slice-rows"); it != settings.options.end())
	{
		try
		{
			int r = std::stoi(it->second);
			if (r >= 1)
				slice_ctb_rows = r;
		}
		catch (...)
		{
		}
	}

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
	if (use_h264)
	{
		// Direct sampling: no copy. The compositor cycles a fixed pool of images, so
		// cache the per-eye R8 / R8G8 plane views PER image (never destroy a view an
		// in-flight encode may still be reading), record this frame's views + the
		// compositor's ready-semaphore for encode() to wait on and sample directly.
		VkImage vkimg = static_cast<VkImage>(y_cbcr);
		auto it = h264_views.find(vkimg);
		if (it == h264_views.end())
		{
			vk::ImageViewUsageCreateInfo usage{.usage = vk::ImageUsageFlagBits::eStorage};
			auto mkview = [&](vk::Format fmt, vk::ImageAspectFlagBits aspect) {
				return vk::raii::ImageView(vk.device, vk::ImageViewCreateInfo{
				        .pNext = &usage, .image = y_cbcr, .viewType = vk::ImageViewType::e2D, .format = fmt,
				        .subresourceRange = {.aspectMask = aspect, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = stream_idx, .layerCount = 1}});
			};
			it = h264_views.emplace(vkimg, std::pair{mkview(vk::Format::eR8Unorm, vk::ImageAspectFlagBits::ePlane0),
			                                         mkview(vk::Format::eR8G8Unorm, vk::ImageAspectFlagBits::ePlane1)})
			             .first;
		}
		in[slot].view_y = *it->second.first;
		in[slot].view_c = *it->second.second;
		in[slot].sem = static_cast<VkSemaphore>(compositor_sem.semaphore);
		in[slot].sem_val = compositor_sem.value;
		return;
	}

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
	if (use_h264)
	{
		// Direct sampling: the recon compute reads the compositor image's plane views
		// directly (no copy), gated on its ready-semaphore, and blocks until done —
		// so the compositor may reuse the image once encode() returns.
		std::shared_ptr<std::vector<uint8_t>> frame;
		{
			std::unique_lock lock(*enc_queue_mutex);
			frame = std::make_shared<std::vector<uint8_t>>(
			        h264_enc.encode_frame_image(h264_cfg, extent.width, extent.height,
			                                    in[slot].view_y, in[slot].view_c, in[slot].sem, in[slot].sem_val));
		}
		const auto & rt = h264_enc.last_timings;
		prof.recon += rt.recon_us;
		prof.cabac += rt.cavlc_us;
		prof.build += rt.assemble_us;
		prof.up += rt.upload;
		prof.bytes += (double)frame->size();
		static const bool profile_enabled = std::getenv("WIVRN_H267_PROFILE") != nullptr;
		if (profile_enabled && ++prof.n >= prof_window)
		{
			const double n = prof.n;
			U_LOG_W("h2-67/h264[%u] %.0fx%u avg/frame: up=%.0fus recon=%.0fus cavlc(gpu)=%.0fus asm=%.0fus | total=%.2fms frame=%.0fKB",
			        stream_idx, (double)extent.width, extent.height, prof.up / n, prof.recon / n, prof.cabac / n, prof.build / n,
			        (prof.up + prof.recon + prof.cabac + prof.build) / n / 1000.0, prof.bytes / n / 1024.0);
			prof = {};
		}
		(void)frame_index;
		return data{.encoder = this, .span = std::span<uint8_t>(*frame), .mem = frame, .prefer_control = true};
	}

	// HEVC path: wait the copy fence, then reconstruct from the mapped host planes.
	if (vk.device.waitForFences(*in[slot].fence, true, 1'000'000'000) == vk::Result::eTimeout)
	{
		U_LOG_E("Timeout on stream %d", stream_idx);
		return {};
	}
	const uint8_t * luma = reinterpret_cast<const uint8_t *>(in[slot].luma.map());
	const uint8_t * chroma = reinterpret_cast<const uint8_t *>(in[slot].chroma.map());

	// Full GPU path: reconstruction + CABAC entirely on the GPU (the level buffers
	// never leave the device); only the compressed per-slice payloads come back.
	{
		std::unique_lock lock(*enc_queue_mutex);
		recon.encode_frame(cfg, extent.width, extent.height, luma, chroma,
		                   slice_ctb_rows, slice_payloads);
	}

	auto t_asm0 = std::chrono::steady_clock::now();
	// Assemble the frame: parameter sets + one IDR NAL per slice (header + payload).
	auto frame = std::make_shared<std::vector<uint8_t>>();
	frame->insert(frame->end(), parameter_sets.begin(), parameter_sets.end());
	const uint32_t nx = cfg.ctbs_x();
	size_t total_payload = 0;
	for (size_t s = 0; s < slice_payloads.size(); ++s)
	{
		h267::bitwriter hdr;
		h267::write_slice_header(hdr, cfg, (uint32_t)(s * slice_ctb_rows) * nx, s == 0);
		std::vector<uint8_t> rbsp = hdr.bytes();
		rbsp.insert(rbsp.end(), slice_payloads[s].begin(), slice_payloads[s].end());
		std::vector<uint8_t> nal;
		h267::emit_nal(nal, h267::NAL_IDR_W_RADL, rbsp);
		frame->insert(frame->end(), nal.begin(), nal.end());
		total_payload += slice_payloads[s].size();
	}

	// Note: the base class writes the returned span to WIVRN_DUMP_VIDEO in SendData().

	// Rolling per-stage profile. encode_frame's last_timings: upload / wavefronts /
	// cabac / payload-readback.
	using clk = std::chrono::steady_clock;
	auto us = [](clk::time_point a, clk::time_point b) {
		return std::chrono::duration<double, std::micro>(b - a).count();
	};
	const auto & rt = recon.last_timings;
	prof.up += rt.upload_us;
	prof.recon += rt.luma_us;    // reconstruction wavefronts (luma+chroma)
	prof.cabac += rt.chroma_us;  // GPU CABAC
	prof.rb += rt.readback_us;   // payload readback
	prof.build += us(t_asm0, clk::now()); // CPU NAL assembly
	prof.bytes += (double)total_payload;
	// Opt-in per-stage profiling: set WIVRN_H267_PROFILE=1 to log stage averages.
	static const bool profile_enabled = std::getenv("WIVRN_H267_PROFILE") != nullptr;
	if (profile_enabled && ++prof.n >= prof_window)
	{
		const double n = prof.n;
		U_LOG_W("h2-67[%u] %.0fx%u %zu slices avg/frame: up=%.0fus recon=%.0fus cabac(gpu)=%.0fus readback=%.0fus asm=%.0fus | total=%.2fms frame=%.0fKB",
		        stream_idx, (double)extent.width, extent.height, slice_payloads.size(),
		        prof.up / n, prof.recon / n, prof.cabac / n, prof.rb / n, prof.build / n,
		        (prof.up + prof.recon + prof.cabac + prof.rb + prof.build) / n / 1000.0, prof.bytes / n / 1024.0);
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
