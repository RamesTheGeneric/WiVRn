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

#include "cabac_pass.h"

#include "cabac_reference.h"
#include "residual_coding.h"

#include <vector>

namespace wivrn::h267
{

using namespace tables;

namespace
{

constexpr int MODE_PLANAR = 0, MODE_DC = 1, MODE_VER = 26;

// Serial entropy coder driven by precomputed per-CU syntax. Mirrors the CABAC
// emission of cpu_encoder exactly (split tree, part_mode, MPM, cbf, residual)
// but performs no reconstruction — it reads modes/cbf/levels from block_syntax.
struct slice_coder
{
	const hevc_config & cfg;
	const block_syntax & bs;
	int bw, bh, w;
	cabac_encoder cb;
	std::vector<uint8_t> mode_plane; // per luma pixel
	std::vector<uint8_t> ct_depth;   // per luma pixel
	std::vector<uint8_t> avail;      // per luma pixel (decoded before)

	slice_coder(const hevc_config & c, const block_syntax & b) :
	        cfg(c), bs(b)
	{
		w = cfg.coded_width();
		bw = w / 8;
		bh = cfg.coded_height() / 8;
		mode_plane.assign((size_t)w * cfg.coded_height(), MODE_DC);
		ct_depth.assign((size_t)w * cfg.coded_height(), 0);
		avail.assign((size_t)w * cfg.coded_height(), 0);
	}

	bool ok(int x, int y) const
	{
		return x >= 0 && y >= 0 && x < w && y < cfg.coded_height() && avail[(size_t)y * w + x];
	}

	void build_mpm(int x0, int y0, int * cand)
	{
		auto nb = [&](int nx, int ny, bool ctb) -> int {
			if (ctb && ny < ((y0 >> 6) << 6))
				return MODE_DC;
			if (nx < 0 || ny < 0 || not avail[(size_t)ny * w + nx])
				return MODE_DC;
			return mode_plane[(size_t)ny * w + nx];
		};
		const int a = nb(x0 - 1, y0, false);
		const int b = nb(x0, y0 - 1, true);
		if (a == b)
		{
			if (a < 2)
			{
				cand[0] = MODE_PLANAR;
				cand[1] = MODE_DC;
				cand[2] = MODE_VER;
			}
			else
			{
				cand[0] = a;
				cand[1] = 2 + ((a + 29) % 32);
				cand[2] = 2 + ((a - 2 + 1) % 32);
			}
		}
		else
		{
			cand[0] = a;
			cand[1] = b;
			if (a != MODE_PLANAR && b != MODE_PLANAR)
				cand[2] = MODE_PLANAR;
			else if (a != MODE_DC && b != MODE_DC)
				cand[2] = MODE_DC;
			else
				cand[2] = MODE_VER;
		}
	}

	void process_cu(int x0, int y0)
	{
		const int b = (y0 / 8) * bw + (x0 / 8);
		const int mode = bs.mode[b];
		const int cbf_luma = bs.cbf_luma[b], cbf_cb = bs.cbf_cb[b], cbf_cr = bs.cbf_cr[b];

		for (int j = 0; j < 8; ++j)
			for (int i = 0; i < 8; ++i)
			{
				mode_plane[(size_t)(y0 + j) * w + (x0 + i)] = (uint8_t)mode;
				ct_depth[(size_t)(y0 + j) * w + (x0 + i)] = 3;
			}

		cb.encode_bin(PART_MODE + 0, 1); // 2Nx2N

		int cand[3];
		build_mpm(x0, y0, cand);
		int mpm_idx = 0;
		while (mpm_idx < 3 && cand[mpm_idx] != mode)
			++mpm_idx;
		cb.encode_bin(PREV_INTRA_LUMA_PRED_FLAG, 1);
		cb.encode_bypass(mpm_idx > 0 ? 1 : 0);
		if (mpm_idx > 0)
			cb.encode_bypass(mpm_idx > 1 ? 1 : 0);

		cb.encode_bin(INTRA_CHROMA_PRED_MODE + 0, 0); // DM

		cb.encode_bin(CBF_CB_CR + 0, cbf_cb);
		cb.encode_bin(CBF_CB_CR + 0, cbf_cr);
		cb.encode_bin(CBF_LUMA + 1, cbf_luma);

		if (cbf_luma)
			residual_coding(cb, &bs.lev_y[(size_t)b * 64], 3, 0);
		if (cbf_cb)
			residual_coding(cb, &bs.lev_cb[(size_t)b * 16], 2, 1);
		if (cbf_cr)
			residual_coding(cb, &bs.lev_cr[(size_t)b * 16], 2, 2);

		// Mark this CU decoded (for later neighbours' availability).
		for (int j = 0; j < 8; ++j)
			for (int i = 0; i < 8; ++i)
				avail[(size_t)(y0 + j) * w + (x0 + i)] = 1;
	}

	void quadtree(int x0, int y0, int log2CbSize, int depth)
	{
		if (log2CbSize > 3)
		{
			const bool condL = ok(x0 - 1, y0) && ct_depth[(size_t)y0 * w + (x0 - 1)] > depth;
			const bool condA = ok(x0, y0 - 1) && ct_depth[(size_t)(y0 - 1) * w + x0] > depth;
			cb.encode_bin(SPLIT_CODING_UNIT_FLAG + (condL ? 1 : 0) + (condA ? 1 : 0), 1);
			const int half = 1 << (log2CbSize - 1);
			quadtree(x0, y0, log2CbSize - 1, depth + 1);
			quadtree(x0 + half, y0, log2CbSize - 1, depth + 1);
			quadtree(x0, y0 + half, log2CbSize - 1, depth + 1);
			quadtree(x0 + half, y0 + half, log2CbSize - 1, depth + 1);
		}
		else
		{
			process_cu(x0, y0);
		}
	}

	// CABAC payload only (no header/NAL) for CTB rows [row0, row0+rows).
	std::vector<uint8_t> code(int row0, int rows, bool last)
	{
		cb.init(0, cfg.qp);
		const uint32_t nx = cfg.ctbs_x();
		const uint32_t cy0 = row0, cy1 = row0 + rows;
		for (uint32_t cy = cy0; cy < cy1; ++cy)
			for (uint32_t cx = 0; cx < nx; ++cx)
			{
				quadtree(cx * 64, cy * 64, 6, 0);
				const bool pic_end = last && (cy + 1 == cy1) && (cx + 1 == nx);
				cb.encode_terminate(pic_end ? 1 : 0);
			}
		return cb.finish();
	}

	std::vector<uint8_t> run()
	{
		bitwriter hdr;
		write_slice_header(hdr, cfg, 0, true);
		auto rbsp = hdr.bytes();
		auto cbytes = code(0, (int)cfg.ctbs_y(), true);
		rbsp.insert(rbsp.end(), cbytes.begin(), cbytes.end());

		std::vector<uint8_t> nal;
		emit_nal(nal, NAL_IDR_W_RADL, rbsp);
		return nal;
	}
};

} // namespace

std::vector<uint8_t> encode_slice_from_syntax(const hevc_config & cfg, const block_syntax & bs)
{
	slice_coder sc(cfg, bs);
	return sc.run();
}

std::vector<uint8_t> encode_slice_payload(const hevc_config & cfg, const block_syntax & bs,
                                          int ctb_row0, int ctb_rows, bool last)
{
	slice_coder sc(cfg, bs);
	return sc.code(ctb_row0, ctb_rows, last);
}

} // namespace wivrn::h267
