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

#include "gpu_reconstruct.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace wivrn::h267::gpu
{

namespace
{
int chroma_qp(int q)
{
	if (q < 30)
		return q;
	if (q > 43)
		return q - 6;
	static const int t[14] = {29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37};
	return t[q - 30];
}

#pragma pack(push, 1)
struct push_const
{
	uint32_t w, h; // coded plane dims
	int32_t qp, bd;
	uint32_t diag, bx_start;
	uint32_t ew, eh; // extent (source) dims, this plane's samples
	uint32_t comp;   // chroma component: 0=Cb, 1=Cr (luma ignores)
	uint32_t slice_rows_px; // slice height in this plane's pixels (edge-clamps intra prediction to the slice)
};
#pragma pack(pop)

// Per-diagonal wavefront steps for a bw x bh block grid. ew/eh are the extent
// source dims (for in-shader edge-clamp), comp selects the interleaved chroma
// component, slice_rows_px is the slice band height (this plane's pixels).
std::vector<vk_compute::step> wavefront_steps(int W, int H, int qp, int bw, int bh,
                                              int ew, int eh, int comp, int slice_rows_px)
{
	std::vector<vk_compute::step> steps;
	for (int d = 0; d <= bw + bh - 2; ++d)
	{
		int bxs = std::max(0, d - (bh - 1));
		int bxe = std::min(d, bw - 1);
		push_const pcv{(uint32_t)W, (uint32_t)H, qp, 8, (uint32_t)d, (uint32_t)bxs,
		               (uint32_t)ew, (uint32_t)eh, (uint32_t)comp, (uint32_t)slice_rows_px};
		vk_compute::step s;
		s.gx = (uint32_t)(bxe - bxs + 1);
		s.gy = 1;
		s.gz = 1;
		s.push.resize(sizeof(pcv));
		std::memcpy(s.push.data(), &pcv, sizeof(pcv));
		steps.push_back(std::move(s));
	}
	return steps;
}

#pragma pack(push, 1)
struct cabac_pc
{
	int32_t qp;
	uint32_t bw, bh, ctbs_x, ctbs_y, slice_ctb_rows, out_stride;
};
#pragma pack(pop)
} // namespace

void reconstructor::init_adopt(VkPhysicalDevice phys, VkDevice dev, VkQueue queue, uint32_t qfam,
                               const uint32_t * luma_spv, size_t luma_words,
                               const uint32_t * chroma_spv, size_t chroma_words)
{
	vkc.adopt(phys, dev, queue, qfam);
	pl = vkc.make_pipeline_from_code(luma_spv, luma_words, 4, sizeof(push_const));
	pc = vkc.make_pipeline_from_code(chroma_spv, chroma_words, 4, sizeof(push_const));
	ready = true;
}

void reconstructor::init_cabac_adopt(const uint32_t * cabac_spv, size_t cabac_words)
{
	pcab = vkc.make_pipeline_from_code(cabac_spv, cabac_words, 9, sizeof(cabac_pc));
	cabac_ready = true;
}

void reconstructor::init_cabac_own(const char * cabac_spv_path)
{
	pcab = vkc.make_pipeline(cabac_spv_path, 9, sizeof(cabac_pc));
	cabac_ready = true;
}

void reconstructor::init_own(const char * luma_spv_path, const char * chroma_spv_path)
{
	vkc.init();
	pl = vkc.make_pipeline(luma_spv_path, 4, sizeof(push_const));
	pc = vkc.make_pipeline(chroma_spv_path, 4, sizeof(push_const));
	ready = true;
}

void reconstructor::ensure_buffers(int cw, int ch, int ew, int eh)
{
	if (alloc_cw == cw && alloc_ch == ch)
		return;
	for (auto * b: {&sY, &rY, &lY, &cY, &sChroma, &rCb, &lCb, &cCb, &rCr, &lCr, &cCr})
		vkc.destroy_buffer(*b);

	const int cw2 = cw / 2, ch2 = ch / 2, bw = cw / 8, bh = ch / 8, nb = bw * bh;
	auto round4 = [](size_t n) { return (n + 3) & ~size_t(3); };
	// Source buffers hold the raw uint8 compositor planes (packed 4 bytes/uint),
	// sized to the extent, not the coded size: the shaders edge-clamp-sample them,
	// so the CPU no longer pads/widens. Written by the CPU then read by the GPU ->
	// keep write-combined. Everything the CPU reads back (levels l*, cbf c*, recon
	// r*) is cached so those reads run at cache speed, not ~100 MB/s WC.
	sY = vkc.make_buffer(round4((size_t)ew * eh));            // luma: 1 byte/sample
	sChroma = vkc.make_buffer(round4((size_t)ew * eh / 2));   // CbCr interleaved: 2 bytes/pair
	rY = vkc.make_buffer((size_t)cw * ch * 4, /*cached=*/true);
	lY = vkc.make_buffer((size_t)nb * 64 * 4, /*cached=*/true);
	cY = vkc.make_buffer((size_t)nb * 4, /*cached=*/true);
	for (auto * s: {&rCb, &rCr})
		*s = vkc.make_buffer((size_t)cw2 * ch2 * 4, /*cached=*/true);
	for (auto * s: {&lCb, &lCr})
		*s = vkc.make_buffer((size_t)nb * 16 * 4, /*cached=*/true);
	for (auto * s: {&cCb, &cCr})
		*s = vkc.make_buffer((size_t)nb * 4, /*cached=*/true);
	alloc_cw = cw;
	alloc_ch = ch;
}

void reconstructor::reconstruct(const hevc_config & cfg,
                                int ew, int eh,
                                const uint8_t * lumaU8, const uint8_t * chromaU8,
                                block_syntax & bs,
                                uint8_t * recY, uint8_t * recCb, uint8_t * recCr,
                                int slice_ctb_rows)
{
	using clk = std::chrono::steady_clock;
	auto us = [](clk::time_point a, clk::time_point b) {
		return std::chrono::duration<double, std::micro>(b - a).count();
	};

	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	const int cw2 = cw / 2, ch2 = ch / 2, bw = cw / 8, bh = ch / 8, nb = bw * bh;
	const int ew2 = ew / 2, eh2 = eh / 2;
	// Slice band height in pixels; <=0 means one slice for the whole frame.
	const int R = slice_ctb_rows > 0 ? slice_ctb_rows : (int)cfg.ctbs_y();
	const int slice_px_luma = R * 64;
	const int slice_px_chroma = R * 32;
	ensure_buffers(cw, ch, ew, eh);

	auto t0 = clk::now();
	// Upload the raw compositor planes verbatim (no CPU pad/de-interleave); the
	// shaders edge-clamp-sample them up to the coded size.
	std::memcpy(sY.ptr, lumaU8, (size_t)ew * eh);
	std::memcpy(sChroma.ptr, chromaU8, (size_t)ew * eh / 2);

	auto t1 = clk::now();
	vkc.run_wavefront(pl, {&sY, &rY, &lY, &cY}, wavefront_steps(cw, ch, cfg.qp, bw, bh, ew, eh, 0, slice_px_luma));
	auto t2 = clk::now();
	const int qc = chroma_qp(cfg.qp);
	// Both chroma passes read the one interleaved source; comp selects Cb/Cr.
	vkc.run_wavefront(pc, {&sChroma, &rCb, &lCb, &cCb}, wavefront_steps(cw2, ch2, qc, bw, bh, ew2, eh2, 0, slice_px_chroma));
	vkc.run_wavefront(pc, {&sChroma, &rCr, &lCr, &cCr}, wavefront_steps(cw2, ch2, qc, bw, bh, ew2, eh2, 1, slice_px_chroma));
	auto t3 = clk::now();

	bs.mode.assign(nb, 1); // DC
	bs.cbf_luma.resize(nb);
	bs.cbf_cb.resize(nb);
	bs.cbf_cr.resize(nb);
	bs.lev_y.resize((size_t)nb * 64);
	bs.lev_cb.resize((size_t)nb * 16);
	bs.lev_cr.resize((size_t)nb * 16);
	std::memcpy(bs.lev_y.data(), lY.ptr, (size_t)nb * 64 * 4);
	std::memcpy(bs.lev_cb.data(), lCb.ptr, (size_t)nb * 16 * 4);
	std::memcpy(bs.lev_cr.data(), lCr.ptr, (size_t)nb * 16 * 4);
	auto * cyp = (const uint32_t *)cY.ptr;
	auto * cbp = (const uint32_t *)cCb.ptr;
	auto * crp = (const uint32_t *)cCr.ptr;
	for (int i = 0; i < nb; ++i)
	{
		bs.cbf_luma[i] = (uint8_t)cyp[i];
		bs.cbf_cb[i] = (uint8_t)cbp[i];
		bs.cbf_cr[i] = (uint8_t)crp[i];
	}
	auto copy_rec = [](uint8_t * dst, const vk_compute::buffer & b, size_t n) {
		if (!dst)
			return;
		auto * s = (const int32_t *)b.ptr;
		for (size_t i = 0; i < n; ++i)
			dst[i] = (uint8_t)s[i];
	};
	copy_rec(recY, rY, (size_t)cw * ch);
	copy_rec(recCb, rCb, (size_t)cw2 * ch2);
	copy_rec(recCr, rCr, (size_t)cw2 * ch2);
	auto t4 = clk::now();

	last_timings = {us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4)};
}

void reconstructor::ensure_cabac_buffers(int nslices, int nb, uint32_t stride)
{
	if (cab_slices == nslices && cab_stride == stride)
		return;
	for (auto * b: {&bOut, &bLen, &bAvail})
		vkc.destroy_buffer(*b);
	bOut = vkc.make_buffer((size_t)stride * nslices * 4, /*cached=*/true); // 1 byte/uint
	bLen = vkc.make_buffer((size_t)nslices * 4, /*cached=*/true);
	bAvail = vkc.make_buffer((size_t)nb * nslices * 4);
	cab_slices = nslices;
	cab_stride = stride;
}

void reconstructor::encode_frame(const hevc_config & cfg,
                                 int ew, int eh,
                                 const uint8_t * lumaU8, const uint8_t * chromaU8,
                                 int slice_ctb_rows,
                                 std::vector<std::vector<uint8_t>> & slice_payloads)
{
	using clk = std::chrono::steady_clock;
	auto us = [](clk::time_point a, clk::time_point b) {
		return std::chrono::duration<double, std::micro>(b - a).count();
	};

	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	const int cw2 = cw / 2, ch2 = ch / 2, bw = cw / 8, bh = ch / 8;
	const int ew2 = ew / 2, eh2 = eh / 2;
	const int R = slice_ctb_rows > 0 ? slice_ctb_rows : (int)cfg.ctbs_y();
	const int slice_px_luma = R * 64, slice_px_chroma = R * 32;
	const uint32_t nx = cfg.ctbs_x(), ny = cfg.ctbs_y();
	const int nslices = (ny + R - 1) / R;
	ensure_buffers(cw, ch, ew, eh);

	auto t0 = clk::now();
	std::memcpy(sY.ptr, lumaU8, (size_t)ew * eh);
	std::memcpy(sChroma.ptr, chromaU8, (size_t)ew * eh / 2);

	// Reconstruction wavefronts fill the GPU level/cbf buffers (no host readback).
	auto t1 = clk::now();
	vkc.run_wavefront(pl, {&sY, &rY, &lY, &cY}, wavefront_steps(cw, ch, cfg.qp, bw, bh, ew, eh, 0, slice_px_luma));
	const int qc = chroma_qp(cfg.qp);
	vkc.run_wavefront(pc, {&sChroma, &rCb, &lCb, &cCb}, wavefront_steps(cw2, ch2, qc, bw, bh, ew2, eh2, 0, slice_px_chroma));
	vkc.run_wavefront(pc, {&sChroma, &rCr, &lCr, &cCr}, wavefront_steps(cw2, ch2, qc, bw, bh, ew2, eh2, 1, slice_px_chroma));
	auto t2 = clk::now();

	// GPU CABAC reads the recon's level/cbf buffers directly. Per-slice byte cap =
	// one slice's pixel count (compressed is well under raw) plus margin.
	const uint32_t stride = (uint32_t)cw * (uint32_t)(R * 64) + 4096u;
	ensure_cabac_buffers(nslices, bw * bh, stride);
	cabac_pc pcv{cfg.qp, (uint32_t)bw, (uint32_t)bh, nx, ny, (uint32_t)R, stride};
	vkc.run(pcab, {&cY, &cCb, &cCr, &lY, &lCb, &lCr, &bOut, &bLen, &bAvail},
	        (uint32_t)nslices, 1, 1, &pcv, sizeof(pcv));
	auto t3 = clk::now();

	// Read back the compressed per-slice payloads (small).
	slice_payloads.resize(nslices);
	const uint32_t * lens = (const uint32_t *)bLen.ptr;
	const uint32_t * obytes = (const uint32_t *)bOut.ptr;
	for (int s = 0; s < nslices; ++s)
	{
		const uint32_t len = lens[s];
		auto & p = slice_payloads[s];
		p.resize(len);
		const uint32_t * src = obytes + (size_t)s * stride;
		for (uint32_t i = 0; i < len; ++i)
			p[i] = (uint8_t)(src[i] & 0xff);
	}
	auto t4 = clk::now();

	last_timings = {us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4)};
}

} // namespace wivrn::h267::gpu
