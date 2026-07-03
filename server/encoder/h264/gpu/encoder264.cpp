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
struct RPC { uint32_t mbw, mbh; int32_t qp, qpc; uint32_t cw, ch, ew, eh, diag, mbx_start; };
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
	recon = vkc.make_pipeline_from_code(recon_spv, recon_words, 11, sizeof(RPC));
	emit = vkc.make_pipeline_from_code(emit_spv, emit_words, 8, sizeof(EPC));
	prefix = vkc.make_pipeline_from_code(prefix_spv, prefix_words, 3, sizeof(PPC));
	stitch = vkc.make_pipeline_from_code(stitch_spv, stitch_words, 4, sizeof(SPC));
	ready = true;
}

void encoder::init_own(const char * recon_path, const char * emit_path, const char * prefix_path, const char * stitch_path)
{
	vkc.init();
	recon = vkc.make_pipeline(recon_path, 11, sizeof(RPC));
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
	for (auto * b : {&sY, &sChroma, &rY, &rCb, &rCr, &lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC, &scratch, &bitLen, &offset, &total, &outbits})
		vkc.destroy_buffer(*b);

	const int cw2 = cw / 2, ch2 = ch / 2;
	const int nmb = (cw / 16) * (ch / 16);
	auto round4 = [](size_t n) { return (n + 3) & ~size_t(3); };
	sY = vkc.make_buffer(round4((size_t)ew * eh));
	sChroma = vkc.make_buffer(round4((size_t)ew * eh / 2));
	rY = vkc.make_buffer((size_t)cw * ch * 4, true);
	rCb = vkc.make_buffer((size_t)cw2 * ch2 * 4, true);
	rCr = vkc.make_buffer((size_t)cw2 * ch2 * 4, true);
	lDC = vkc.make_buffer((size_t)nmb * 16 * 4, true);
	lAC = vkc.make_buffer((size_t)nmb * 256 * 4, true);
	cDC = vkc.make_buffer((size_t)nmb * 8 * 4, true);
	cAC = vkc.make_buffer((size_t)nmb * 128 * 4, true);
	nnzL = vkc.make_buffer((size_t)(cw / 4) * (ch / 4) * 4, true);
	nnzC = vkc.make_buffer((size_t)2 * (cw / 8) * (ch / 8) * 4, true);
	scratch = vkc.make_buffer((size_t)nmb * stride_words * 4, true);
	bitLen = vkc.make_buffer((size_t)nmb * 4, true);
	offset = vkc.make_buffer((size_t)nmb * 4, true);
	total = vkc.make_buffer(4, true);
	outbits = vkc.make_buffer((size_t)nmb * stride_words * 4, true);
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
	// reconstruction wavefront steps (anti-diagonal)
	std::vector<vk_compute::buffer *> rbind = {&sY, &sChroma, &rY, &rCb, &rCr, &lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC};
	std::vector<vk_compute::step> rsteps;
	for (int d = 0; d <= mbw + mbh - 2; ++d)
	{
		int s = std::max(0, d - (mbh - 1)), e = std::min(d, mbw - 1);
		RPC pc{(uint32_t)mbw, (uint32_t)mbh, cfg.qp, qpc, (uint32_t)cw, (uint32_t)ch, (uint32_t)ew, (uint32_t)eh, (uint32_t)d, (uint32_t)s};
		vk_compute::step st; st.gx = (uint32_t)(e - s + 1); st.gy = 1; st.gz = 1;
		st.push.resize(sizeof(pc)); std::memcpy(st.push.data(), &pc, sizeof(pc)); rsteps.push_back(std::move(st));
	}

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
	auto b = vkc.begin_batch();
	vkc.record_fill(b, outbits, (size_t)nmb * stride_words * 4, 0);
	vkc.record_wavefront(b, recon, rbind, rsteps, /*leading_barrier=*/false);
	vkc.record_dispatch(b, emit, {&lDC, &lAC, &cDC, &cAC, &nnzL, &nnzC, &scratch, &bitLen}, groups, 1, 1, &epc, sizeof(epc), true);
	vkc.record_dispatch(b, prefix, {&bitLen, &offset, &total}, 1, 1, 1, &ppc, sizeof(ppc), true);
	vkc.record_dispatch(b, stitch, {&scratch, &bitLen, &offset, &outbits}, groups, 1, 1, &spc, sizeof(spc), true);
	vkc.submit_and_wait(b);
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
		auto * s = (const int32_t *)b.ptr;
		for (size_t i = 0; i < n; ++i) dst[i] = (uint8_t)s[i];
	};
	copy_rec(recY, rY, (size_t)cw * ch);
	copy_rec(recCb, rCb, (size_t)cw2 * ch2);
	copy_rec(recCr, rCr, (size_t)cw2 * ch2);

	// up = upload; recon_us = whole GPU batch (recon + CAVLC); cavlc_us folded in; asm = CPU.
	last_timings = {us(t0, t1), us(t1, t2), us(t2, t3), us(t3, clk::now())};
	return frame;
}

} // namespace wivrn::avc::gpu
