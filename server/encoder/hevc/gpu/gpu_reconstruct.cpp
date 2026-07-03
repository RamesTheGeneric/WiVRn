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
#include <cstring>

namespace wivrn::hevc::gpu
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
	uint32_t w, h;
	int32_t qp, bd;
	uint32_t diag, bx_start;
};
#pragma pack(pop)

// Per-diagonal wavefront steps for a bw x bh block grid.
std::vector<vk_compute::step> wavefront_steps(int W, int H, int qp, int bw, int bh)
{
	std::vector<vk_compute::step> steps;
	for (int d = 0; d <= bw + bh - 2; ++d)
	{
		int bxs = std::max(0, d - (bh - 1));
		int bxe = std::min(d, bw - 1);
		push_const pcv{(uint32_t)W, (uint32_t)H, qp, 8, (uint32_t)d, (uint32_t)bxs};
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

void reconstructor::init_own(const char * luma_spv_path, const char * chroma_spv_path)
{
	vkc.init();
	pl = vkc.make_pipeline(luma_spv_path, 4, sizeof(push_const));
	pc = vkc.make_pipeline(chroma_spv_path, 4, sizeof(push_const));
	ready = true;
}

void reconstructor::ensure_buffers(int cw, int ch)
{
	if (alloc_cw == cw && alloc_ch == ch)
		return;
	for (auto * b: {&sY, &rY, &lY, &cY, &sCb, &rCb, &lCb, &cCb, &sCr, &rCr, &lCr, &cCr})
		vkc.destroy_buffer(*b);

	const int cw2 = cw / 2, ch2 = ch / 2, bw = cw / 8, bh = ch / 8, nb = bw * bh;
	sY = vkc.make_buffer((size_t)cw * ch * 4);
	rY = vkc.make_buffer((size_t)cw * ch * 4);
	lY = vkc.make_buffer((size_t)nb * 64 * 4);
	cY = vkc.make_buffer((size_t)nb * 4);
	for (auto * s: {&sCb, &sCr, &rCb, &rCr})
		*s = vkc.make_buffer((size_t)cw2 * ch2 * 4);
	for (auto * s: {&lCb, &lCr})
		*s = vkc.make_buffer((size_t)nb * 16 * 4);
	for (auto * s: {&cCb, &cCr})
		*s = vkc.make_buffer((size_t)nb * 4);
	alloc_cw = cw;
	alloc_ch = ch;
}

void reconstructor::reconstruct(const hevc_config & cfg,
                                const int32_t * srcY, const int32_t * srcCb, const int32_t * srcCr,
                                block_syntax & bs,
                                uint8_t * recY, uint8_t * recCb, uint8_t * recCr)
{
	const int cw = cfg.coded_width(), ch = cfg.coded_height();
	const int cw2 = cw / 2, ch2 = ch / 2, bw = cw / 8, bh = ch / 8, nb = bw * bh;
	ensure_buffers(cw, ch);

	std::memcpy(sY.ptr, srcY, (size_t)cw * ch * 4);
	std::memcpy(sCb.ptr, srcCb, (size_t)cw2 * ch2 * 4);
	std::memcpy(sCr.ptr, srcCr, (size_t)cw2 * ch2 * 4);

	vkc.run_wavefront(pl, {&sY, &rY, &lY, &cY}, wavefront_steps(cw, ch, cfg.qp, bw, bh));
	const int qc = chroma_qp(cfg.qp);
	auto chroma_steps = wavefront_steps(cw2, ch2, qc, bw, bh);
	vkc.run_wavefront(pc, {&sCb, &rCb, &lCb, &cCb}, chroma_steps);
	vkc.run_wavefront(pc, {&sCr, &rCr, &lCr, &cCr}, chroma_steps);

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
}

} // namespace wivrn::hevc::gpu
