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

#include "encoder264.h"

#include "../transform264.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace wivrn::avc::gpu
{

namespace
{
#pragma pack(push, 1)
struct RPC { uint32_t mbw, mbh; int32_t qp, qpc; uint32_t cw, ch, ew, eh, nmb; };
struct EPC { uint32_t mbw, mbh, stride_words, lgw, cgw, cgh; };
struct PPC { uint32_t nmb, base_bits; };
struct SPC { uint32_t nmb, stride_words; };
#pragma pack(pop)
} // namespace

void encoder::init_adopt(VkPhysicalDevice phys, VkDevice dev, VkQueue queue, uint32_t qfam,
                         const uint32_t * recon_spv, size_t recon_words,
                         const uint32_t * emit_spv, size_t emit_words,
                         const uint32_t * prefix_spv, size_t prefix_words,
                         const uint32_t * stitch_spv, size_t stitch_words)
{
	vkc.adopt(phys, dev, queue, qfam);
	recon = vkc.make_pipeline_from_code(recon_spv, recon_words, 15, sizeof(RPC));
	emit = vkc.make_pipeline_from_code(emit_spv, emit_words, 8, sizeof(EPC));
	prefix = vkc.make_pipeline_from_code(prefix_spv, prefix_words, 3, sizeof(PPC));
	stitch = vkc.make_pipeline_from_code(stitch_spv, stitch_words, 4, sizeof(SPC));
	ready = true;
}

void encoder::init_own(const char * recon_path, const char * emit_path, const char * prefix_path, const char * stitch_path)
{
	vkc.init();
	recon = vkc.make_pipeline(recon_path, 15, sizeof(RPC));
	emit = vkc.make_pipeline(emit_path, 8, sizeof(EPC));
	prefix = vkc.make_pipeline(prefix_path, 3, sizeof(PPC));
	stitch = vkc.make_pipeline(stitch_path, 4, sizeof(SPC));
	ready = true;
}

void encoder::ensure_buffers(const h264_config & cfg, int ew, int eh)
{
	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	if (alloc_cw == cw && alloc_ch == ch && alloc_ew == ew)
		return;
	for (auto * b : {&sY, &sChroma, &rY, &rCb, &rCr, &lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC, &scratch, &bitLen, &offset, &total, &outbits, &mbOrder, &claim, &doneBuf, &haloBuf})
		vkc.destroy_buffer(*b);

	const int cw2 = cw / 2, ch2 = ch / 2;
	const int nmb = (cw / 16) * (ch / 16);
	auto round4 = [](size_t n) { return (n + 3) & ~size_t(3); };
	// GPU-internal buffers (never touched by the CPU in the live path) go on
	// device-local memory; only the source upload (WC) and outbits/total (CPU-read
	// each frame, host-cached) stay on host memory.
	const int G = vk_compute::MEM_GPU;
	sY = vkc.make_buffer(round4((size_t)ew * eh));                        // upload (WC)
	sChroma = vkc.make_buffer(round4((size_t)ew * eh / 2));               // upload (WC)
	rY = vkc.make_buffer(round4((size_t)cw * ch), G);      // packed u8
	rCb = vkc.make_buffer(round4((size_t)cw2 * ch2), G);   // packed u8
	rCr = vkc.make_buffer(round4((size_t)cw2 * ch2), G);   // packed u8
	lDC = vkc.make_buffer((size_t)nmb * 16 * 2, G);  // int16 levels
	lAC = vkc.make_buffer((size_t)nmb * 256 * 2, G); // int16 levels
	cDC = vkc.make_buffer((size_t)nmb * 8 * 2, G);   // int16 levels
	cAC = vkc.make_buffer((size_t)nmb * 128 * 2, G); // int16 levels
	nnzL = vkc.make_buffer((size_t)(cw / 4) * (ch / 4) * 4, G);
	nnzC = vkc.make_buffer((size_t)2 * (cw / 8) * (ch / 8) * 4, G);
	scratch = vkc.make_buffer((size_t)nmb * stride_words * 4, G);
	bitLen = vkc.make_buffer((size_t)nmb * 4, G);
	offset = vkc.make_buffer((size_t)nmb * 4, G);
	total = vkc.make_buffer(4, true);                                    // CPU reads total bits
	outbits = vkc.make_buffer((size_t)nmb * stride_words * 4, true);      // CPU reads payload
	// Scoreboard: MB indices in anti-diagonal (dependency) order (written once
	// here on the CPU), a claim counter, and one done flag per MB (both GPU-filled
	// to 0 each frame).
	mbOrder = vkc.make_buffer((size_t)nmb * 4);                          // CPU-written once (WC)
	claim = vkc.make_buffer(4, G);
	doneBuf = vkc.make_buffer((size_t)nmb * 4, G);
	haloBuf = vkc.make_buffer((size_t)nmb * 16 * 4, G);                  // 64B/MB boundary pixels
	{
		const int mbw = cw / 16, mbh = ch / 16;
		auto * ord = static_cast<uint32_t *>(mbOrder.ptr);
		uint32_t k = 0;
		for (int d = 0; d <= mbw + mbh - 2; ++d)
			for (int mbx = std::max(0, d - (mbh - 1)); mbx <= std::min(d, mbw - 1); ++mbx)
				ord[k++] = uint32_t((d - mbx) * mbw + mbx);
	}
	alloc_cw = cw; alloc_ch = ch; alloc_ew = ew;
}

std::vector<uint8_t> encoder::encode_frame(const h264_config & cfg, int ew, int eh,
                                           const uint8_t * lumaU8, const uint8_t * chromaU8,
                                           uint8_t * recY, uint8_t * recCb, uint8_t * recCr)
{
	using clk = std::chrono::steady_clock;
	auto us = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };

	const int cw = cfg.coded_width(), ch = cfg.coded_height(), cw2 = cw / 2, ch2 = ch / 2;
	const int mbw = cfg.mb_width(), mbh = cfg.mb_height(), nmb = mbw * mbh;
	const int qpc = wivrn::avc::xform::chroma_qp(cfg.qp);
	ensure_buffers(cfg, ew, eh);

	auto t0 = clk::now();
	std::memcpy(sY.ptr, lumaU8, (size_t)ew * eh);
	std::memcpy(sChroma.ptr, chromaU8, (size_t)ew * eh / 2);

	auto t1 = clk::now();
	// Scoreboard reconstruction: one dispatch of N persistent workgroups. Each
	// claims MBs from an atomic counter in anti-diagonal (dependency) order
	// (mbOrder) and spins on the left/top neighbours' done flags (device-scoped
	// atomic acquire/release) before running — all cross-MB dependencies in one
	// dispatch instead of ~99 pipeline barriers. N stays <= the co-resident
	// workgroup count for forward progress; 64 is trivially resident on the
	// 16-CU APU and exceeds the max anti-diagonal width (no lost parallelism).
	std::vector<vk_compute::buffer *> rbind = {&sY, &sChroma, &rY, &rCb, &rCr, &lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC, &mbOrder, &claim, &doneBuf, &haloBuf};
	const RPC rpc{(uint32_t)mbw, (uint32_t)mbh, cfg.qp, qpc, (uint32_t)cw, (uint32_t)ch, (uint32_t)ew, (uint32_t)eh, (uint32_t)nmb};
	uint32_t recon_wg_cap = 64u;
	if (const char * e = std::getenv("WIVRN_H264_RECON_WG")) { int w = std::atoi(e); if (w > 0) recon_wg_cap = (uint32_t)w; }
	const uint32_t recon_wg = std::min<uint32_t>((uint32_t)nmb, recon_wg_cap);

	// Build the slice header (tiny) up front to get its exact bit length, which the
	// prefix uses as the payload's base bit offset so the stitch lays the payload
	// down right after the header — no CPU bit-append of the payload.
	bitwriter hdr;
	write_slice_header(hdr, cfg, 0, 0);
	const uint32_t hbits = (uint32_t)hdr.bit_count();  // exact header bit length (not byte-aligned)
	const std::vector<uint8_t> hbytes = hdr.bytes();   // header bits MSB-first, last byte zero-padded

	// One command buffer: clear output, recon wavefront, then CAVLC emit -> prefix
	// -> stitch, with compute barriers between stages. Single submit + fence wait.
	const EPC epc{(uint32_t)mbw, (uint32_t)mbh, stride_words, (uint32_t)(cw / 4), (uint32_t)(cw / 8), (uint32_t)(ch / 8)};
	const PPC ppc{(uint32_t)nmb, hbits};
	const SPC spc{(uint32_t)nmb, stride_words};
	const uint32_t groups = (uint32_t)((nmb + 63) / 64);
	// Opt-in GPU-stage profiling: WIVRN_H264_GPU_TIMESTAMPS=1 splits the merged
	// batch into recon-wavefront vs CAVLC (emit+prefix+stitch) via GPU timestamps.
	static const bool gpu_prof = std::getenv("WIVRN_H264_GPU_TIMESTAMPS") != nullptr;
	vkc.profile_gpu = gpu_prof;
	auto b = vkc.begin_batch();
	vkc.record_timestamp(b); // ts0: batch start (before fills)
	vkc.record_fill(b, outbits, (size_t)nmb * stride_words * 4, 0);
	vkc.record_fill(b, claim, 4, 0);                 // claim counter = 0
	vkc.record_fill(b, doneBuf, (size_t)nmb * 4, 0); // per-MB done flags = 0
	vkc.record_timestamp(b); // ts1: fills done, before recon
	vkc.record_dispatch(b, recon, rbind, recon_wg, 1, 1, &rpc, sizeof(rpc), /*leading_barrier=*/false);
	vkc.record_timestamp(b); // ts2: recon done
	// emit is now one workgroup (32 lanes) per MB: intra-MB parallel over segments.
	vkc.record_dispatch(b, emit, {&lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC, &scratch, &bitLen}, (uint32_t)nmb, 1, 1, &epc, sizeof(epc), true);
	vkc.record_timestamp(b); // ts3: emit done
	vkc.record_dispatch(b, prefix, {&bitLen, &offset, &total}, 1, 1, 1, &ppc, sizeof(ppc), true);
	vkc.record_timestamp(b); // ts4: prefix done
	vkc.record_dispatch(b, stitch, {&scratch, &bitLen, &offset, &outbits}, groups, 1, 1, &spc, sizeof(spc), true);
	vkc.record_timestamp(b); // ts5: stitch done
	vkc.submit_and_wait(b);
	const double gpu_fills_us = vkc.ts_us(b, 0, 1);
	const double gpu_recon_us = vkc.ts_us(b, 1, 2);
	const double gpu_cavlc_us = vkc.ts_us(b, 2, 5);
	const double gpu_emit_us = vkc.ts_us(b, 2, 3);
	const double gpu_prefix_us = vkc.ts_us(b, 3, 4);
	const double gpu_stitch_us = vkc.ts_us(b, 4, 5);
	const uint32_t totbits = ((const uint32_t *)total.ptr)[0]; // header + payload bits

	auto t2 = clk::now();
	auto t3 = t2; // CAVLC merged into the single GPU submit

	// The GPU cleared outbits and stitched the payload at bit offset hbits; bits
	// [0, hbits) are zero. Seed the header bits there (MSB-first) on the CPU (tiny),
	// then the rbsp_trailing stop bit at `totbits`, and extract RBSP bytes.
	uint32_t * ow = (uint32_t *)outbits.ptr;
	auto set_bit = [&](uint32_t pos, uint32_t bit) {
		if (bit)
			ow[pos >> 5] |= (0x80000000u >> (pos & 31));
	};
	for (uint32_t i = 0; i < hbits; ++i)
		set_bit(i, (hbytes[i >> 3] >> (7 - (i & 7))) & 1u);
	set_bit(totbits, 1); // rbsp_trailing_bits stop bit; remaining bits already zero

	const size_t rbsp_bytes = (size_t)(totbits + 1 + 7) / 8;
	std::vector<uint8_t> rbsp(rbsp_bytes);
	for (size_t i = 0; i < rbsp_bytes; ++i)
		rbsp[i] = (uint8_t)((ow[i >> 2] >> (24 - 8 * (i & 3))) & 0xffu);

	std::vector<uint8_t> frame = build_parameter_sets(cfg);
	emit_nal(frame, 3, NAL_SLICE_IDR, rbsp);

	auto copy_rec = [](uint8_t * dst, const vk_compute::buffer & b, size_t n) {
		if (!dst) return;
		std::memcpy(dst, b.ptr, n); // recon planes are packed u8
	};
	copy_rec(recY, rY, (size_t)cw * ch);
	copy_rec(recCb, rCb, (size_t)cw2 * ch2);
	copy_rec(recCr, rCr, (size_t)cw2 * ch2);

	// up = upload (CPU memcpy); asm = CPU final assembly. When GPU timestamps are
	// enabled, recon_us / cavlc_us are the true GPU-side split; otherwise recon_us
	// is the whole batch wall-clock and cavlc_us folds into it (0).
	const double recon_report = gpu_prof ? gpu_recon_us : us(t1, t2);
	const double cavlc_report = gpu_prof ? gpu_cavlc_us : 0.0;
	last_timings = {us(t0, t1), recon_report, cavlc_report, us(t3, clk::now()),
	                gpu_emit_us, gpu_prefix_us, gpu_stitch_us, gpu_fills_us};
	return frame;
}

} // namespace wivrn::avc::gpu
