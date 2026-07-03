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

#include "cpu_encoder.h"

#include "cabac_reference.h"
#include "residual_coding.h"
#include "transform.h"

#include <array>
#include <cstdlib>
#include <vector>

namespace wivrn::h267
{

using namespace tables;

namespace
{

constexpr int MODE_PLANAR = 0;
constexpr int MODE_DC = 1;
constexpr int MODE_VER = 26;

int clip_pixel(int v, int max)
{
	return v < 0 ? 0 : (v > max ? max : v);
}

// Chroma QP derivation for 4:2:0 (Rec. ITU-T H.265 clause 8.6.1, Table 8-10).
// pps/slice chroma QP offsets are 0 in our config, so qPi == luma QP.
int chroma_qp(int luma_qp)
{
	const int qpi = luma_qp;
	if (qpi < 30)
		return qpi;
	if (qpi > 43)
		return qpi - 6;
	static const int tab[14] = {29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37};
	return tab[qpi - 30];
}

// A single image plane with reconstruction, availability, and (luma) side info.
struct plane
{
	std::vector<int> rec;
	std::vector<uint8_t> avail;
	int w = 0, h = 0;
	void init(int width, int height)
	{
		w = width;
		h = height;
		rec.assign((size_t)w * h, 0);
		avail.assign((size_t)w * h, 0);
	}
	int at(int x, int y) const { return rec[(size_t)y * w + x]; }
	bool ok(int x, int y) const { return x >= 0 && y >= 0 && x < w && y < h && avail[(size_t)y * w + x]; }
};

// Gather reference samples for a TU with substitution (8.4.4.2.1/2). refTop and
// refLeft have 2N+1 entries; index 0 is the corner p[-1][-1].
void get_refs(const plane & pl, int xTb, int yTb, int N, int bd, int * refTop, int * refLeft)
{
	const int total = 4 * N + 1;
	std::vector<int> seq(total);
	std::vector<uint8_t> sav(total);
	int idx = 0;
	for (int y = 2 * N - 1; y >= 0; --y) // left column, bottom -> top
	{
		const bool a = pl.ok(xTb - 1, yTb + y);
		sav[idx] = a;
		seq[idx] = a ? pl.at(xTb - 1, yTb + y) : 0;
		++idx;
	}
	{ // corner
		const bool a = pl.ok(xTb - 1, yTb - 1);
		sav[idx] = a;
		seq[idx] = a ? pl.at(xTb - 1, yTb - 1) : 0;
		++idx;
	}
	for (int x = 0; x < 2 * N; ++x) // top row, left -> right
	{
		const bool a = pl.ok(xTb + x, yTb - 1);
		sav[idx] = a;
		seq[idx] = a ? pl.at(xTb + x, yTb - 1) : 0;
		++idx;
	}

	bool any = false;
	for (int i = 0; i < total; ++i)
		any = any || sav[i];
	if (not any)
	{
		const int def = 1 << (bd - 1);
		for (int i = 0; i < total; ++i)
			seq[i] = def;
	}
	else
	{
		if (not sav[0])
		{
			int first = 0;
			while (not sav[first])
				++first;
			for (int i = 0; i < first; ++i)
				seq[i] = seq[first];
		}
		for (int i = 1; i < total; ++i)
			if (not sav[i])
				seq[i] = seq[i - 1];
	}

	refLeft[0] = refTop[0] = seq[2 * N];
	for (int k = 1; k <= 2 * N; ++k)
	{
		refLeft[k] = seq[2 * N - k];
		refTop[k] = seq[2 * N + k];
	}
}

// [1 2 1] reference-sample smoothing (8.4.4.2.3), luma only.
void filter_refs(int * refTop, int * refLeft, int N)
{
	const int corner = refTop[0];
	std::vector<int> nt(2 * N + 1), nl(2 * N + 1);
	nt[0] = nl[0] = (refLeft[1] + 2 * corner + refTop[1] + 2) >> 2;
	for (int k = 1; k < 2 * N; ++k)
	{
		nt[k] = (refTop[k - 1] + 2 * refTop[k] + refTop[k + 1] + 2) >> 2;
		nl[k] = (refLeft[k - 1] + 2 * refLeft[k] + refLeft[k + 1] + 2) >> 2;
	}
	nt[2 * N] = refTop[2 * N];
	nl[2 * N] = refLeft[2 * N];
	for (int k = 0; k <= 2 * N; ++k)
	{
		refTop[k] = nt[k];
		refLeft[k] = nl[k];
	}
}

int log2_of(int n)
{
	int l = 0;
	while ((1 << l) < n)
		++l;
	return l;
}

// DC / Planar prediction into pred[N*N] (row-major). cIdx: 0 = luma.
void predict(int mode, int N, int bd, int cIdx, const int * refTop, const int * refLeft, int * pred)
{
	const int log2n = log2_of(N);
	if (mode == MODE_PLANAR)
	{
		const int topRight = refTop[N + 1]; // p[N][-1]
		const int botLeft = refLeft[N + 1]; // p[-1][N]
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
			{
				const int v = (N - 1 - x) * refLeft[1 + y] + (x + 1) * topRight + (N - 1 - y) * refTop[1 + x] + (y + 1) * botLeft + N;
				pred[y * N + x] = v >> (log2n + 1);
			}
	}
	else // DC
	{
		int sum = 0;
		for (int i = 0; i < N; ++i)
			sum += refTop[1 + i] + refLeft[1 + i];
		const int dc = (sum + N) >> (log2n + 1);
		for (int i = 0; i < N * N; ++i)
			pred[i] = dc;
		if (cIdx == 0 && N < 32) // DC boundary smoothing (8.4.4.2.5)
		{
			pred[0] = (refLeft[1] + 2 * dc + refTop[1] + 2) >> 2;
			for (int x = 1; x < N; ++x)
				pred[x] = (refTop[1 + x] + 3 * dc + 2) >> 2;
			for (int y = 1; y < N; ++y)
				pred[y * N] = (refLeft[1 + y] + 3 * dc + 2) >> 2;
		}
	}
	(void)bd;
}

struct encoder
{
	const hevc_config & cfg;
	const yuv_image & src;
	int bd;
	int max_pix;
	plane Y, Cb, Cr;
	std::vector<uint8_t> mode_plane; // luma intra mode per luma pixel
	std::vector<uint8_t> ct_depth;   // coding tree depth per luma pixel
	cabac_encoder cb;

	encoder(const hevc_config & c, const yuv_image & s) :
	        cfg(c), src(s), bd(c.bit_depth), max_pix((1 << c.bit_depth) - 1)
	{
		Y.init(cfg.coded_width(), cfg.coded_height());
		Cb.init(cfg.coded_width() / 2, cfg.coded_height() / 2);
		Cr.init(cfg.coded_width() / 2, cfg.coded_height() / 2);
		mode_plane.assign((size_t)Y.w * Y.h, MODE_DC);
		ct_depth.assign((size_t)Y.w * Y.h, 0);
	}

	// Encode one component TU: predict, transform, quantise, reconstruct,
	// return cbf and fill `level`.
	int code_tu(plane & pl, const uint16_t * src_plane, int src_w, int x, int y, int N, int mode, int cIdx, int32_t * level)
	{
		int refTop[65], refLeft[65];
		get_refs(pl, x, y, N, bd, refTop, refLeft);

		// Reference filtering for luma when the mode requires it.
		if (cIdx == 0 && mode != MODE_DC && N != 4)
		{
			const int minDist = std::min(std::abs(mode - 26), std::abs(mode - 10));
			const int thres = (N == 8) ? 7 : (N == 16) ? 1 : 0;
			if (minDist > thres)
				filter_refs(refTop, refLeft, N);
		}

		int pred[64];
		predict(mode, N, bd, cIdx, refTop, refLeft, pred);

		int32_t resid[64];
		for (int j = 0; j < N; ++j)
			for (int i = 0; i < N; ++i)
				resid[j * N + i] = (int)src_plane[(size_t)(y + j) * src_w + (x + i)] - pred[j * N + i];

		const int qp = (cIdx == 0) ? cfg.qp : chroma_qp(cfg.qp);
		int32_t coeff[64];
		xform::fdct(resid, coeff, N, bd);
		xform::quant(coeff, level, N, qp, bd);

		int cbf = 0;
		for (int i = 0; i < N * N; ++i)
			cbf |= (level[i] != 0);

		int recon[64];
		if (cbf)
		{
			int32_t deq[64], rres[64];
			xform::dequant(level, deq, N, qp, bd);
			xform::idct(deq, rres, N, bd);
			for (int i = 0; i < N * N; ++i)
				recon[i] = clip_pixel(pred[i] + rres[i], max_pix);
		}
		else
		{
			for (int i = 0; i < N * N; ++i)
				recon[i] = clip_pixel(pred[i], max_pix);
		}

		for (int j = 0; j < N; ++j)
			for (int i = 0; i < N; ++i)
			{
				pl.rec[(size_t)(y + j) * pl.w + (x + i)] = recon[j * N + i];
				pl.avail[(size_t)(y + j) * pl.w + (x + i)] = 1;
			}
		return cbf;
	}

	// Choose DC vs Planar for the luma block by SAD.
	int choose_luma_mode(int x, int y)
	{
		if (cfg.force_luma_mode >= 0)
			return cfg.force_luma_mode;
		int refTop[65], refLeft[65];
		get_refs(Y, x, y, 8, bd, refTop, refLeft);
		int predDC[64];
		predict(MODE_DC, 8, bd, 0, refTop, refLeft, predDC);
		int ftop[65], fleft[65];
		for (int k = 0; k < 65; ++k)
		{
			ftop[k] = refTop[k];
			fleft[k] = refLeft[k];
		}
		filter_refs(ftop, fleft, 8); // planar 8x8 uses filtered refs
		int predPl[64];
		predict(MODE_PLANAR, 8, bd, 0, ftop, fleft, predPl);

		long sadDC = 0, sadPl = 0;
		for (int j = 0; j < 8; ++j)
			for (int i = 0; i < 8; ++i)
			{
				const int s = src.Y[(size_t)(y + j) * Y.w + (x + i)];
				sadDC += std::abs(s - predDC[j * 8 + i]);
				sadPl += std::abs(s - predPl[j * 8 + i]);
			}
		return (sadDC <= sadPl) ? MODE_DC : MODE_PLANAR;
	}

	void build_mpm(int x0, int y0, int * cand)
	{
		auto neighbour_mode = [&](int nx, int ny, bool constrain_ctb) -> int {
			if (constrain_ctb && (ny < ((y0 >> 6) << 6)))
				return MODE_DC; // above neighbour in a different CTB row
			if (nx < 0 || ny < 0 || not Y.avail[(size_t)ny * Y.w + nx])
				return MODE_DC;
			return mode_plane[(size_t)ny * Y.w + nx];
		};
		const int a = neighbour_mode(x0 - 1, y0, false);
		const int b = neighbour_mode(x0, y0 - 1, true);
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
		const int mode = choose_luma_mode(x0, y0);

		int32_t lvlY[64], lvlCb[16], lvlCr[16];
		const int cbf_luma = code_tu(Y, src.Y.data(), Y.w, x0, y0, 8, mode, 0, lvlY);
		const int cbf_cb = code_tu(Cb, src.Cb.data(), Cb.w, x0 / 2, y0 / 2, 4, mode, 1, lvlCb);
		const int cbf_cr = code_tu(Cr, src.Cr.data(), Cr.w, x0 / 2, y0 / 2, 4, mode, 2, lvlCr);

		// Record luma mode and coding-tree depth for neighbour derivation.
		for (int j = 0; j < 8; ++j)
			for (int i = 0; i < 8; ++i)
			{
				mode_plane[(size_t)(y0 + j) * Y.w + (x0 + i)] = mode;
				ct_depth[(size_t)(y0 + j) * Y.w + (x0 + i)] = 3;
			}

		// --- CABAC (decode order) ---
		cb.encode_bin(PART_MODE + 0, 1); // part_mode = 2Nx2N (intra, min CU)

		int cand[3];
		build_mpm(x0, y0, cand);
		int mpm_idx = 0;
		while (mpm_idx < 3 && cand[mpm_idx] != mode)
			++mpm_idx;
		// DC and Planar are always present in the MPM list for this encoder.
		cb.encode_bin(PREV_INTRA_LUMA_PRED_FLAG, 1);
		cb.encode_bypass(mpm_idx > 0 ? 1 : 0);
		if (mpm_idx > 0)
			cb.encode_bypass(mpm_idx > 1 ? 1 : 0);

		cb.encode_bin(INTRA_CHROMA_PRED_MODE + 0, 0); // derived mode (DM)

		cb.encode_bin(CBF_CB_CR + 0, cbf_cb);
		cb.encode_bin(CBF_CB_CR + 0, cbf_cr);
		cb.encode_bin(CBF_LUMA + 1, cbf_luma); // trafoDepth 0 -> ctxInc 1

		if (cbf_luma)
			residual_coding(cb, lvlY, 3, 0);
		if (cbf_cb)
			residual_coding(cb, lvlCb, 2, 1);
		if (cbf_cr)
			residual_coding(cb, lvlCr, 2, 2);
	}

	void quadtree(int x0, int y0, int log2CbSize, int depth)
	{
		if (log2CbSize > 3)
		{
			const bool condL = Y.ok(x0 - 1, y0) && ct_depth[(size_t)y0 * Y.w + (x0 - 1)] > depth;
			const bool condA = Y.ok(x0, y0 - 1) && ct_depth[(size_t)(y0 - 1) * Y.w + x0] > depth;
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

	std::vector<uint8_t> run(yuv_image * recon_out)
	{
		bitwriter hdr;
		write_slice_header(hdr, cfg, 0, true);
		auto rbsp = hdr.bytes();

		cb.init(0, cfg.qp); // I slice

		const uint32_t nx = cfg.ctbs_x(), ny = cfg.ctbs_y();
		for (uint32_t cy = 0; cy < ny; ++cy)
			for (uint32_t cx = 0; cx < nx; ++cx)
			{
				quadtree(cx * 64, cy * 64, 6, 0);
				const bool last = (cy + 1 == ny) && (cx + 1 == nx);
				cb.encode_terminate(last ? 1 : 0);
			}

		auto cbytes = cb.finish();
		rbsp.insert(rbsp.end(), cbytes.begin(), cbytes.end());

		std::vector<uint8_t> nal;
		emit_nal(nal, NAL_IDR_W_RADL, rbsp);

		if (recon_out)
		{
			recon_out->width = Y.w;
			recon_out->height = Y.h;
			recon_out->Y.resize(Y.rec.size());
			recon_out->Cb.resize(Cb.rec.size());
			recon_out->Cr.resize(Cr.rec.size());
			for (size_t i = 0; i < Y.rec.size(); ++i)
				recon_out->Y[i] = (uint16_t)Y.rec[i];
			for (size_t i = 0; i < Cb.rec.size(); ++i)
			{
				recon_out->Cb[i] = (uint16_t)Cb.rec[i];
				recon_out->Cr[i] = (uint16_t)Cr.rec[i];
			}
		}
		return nal;
	}
};

} // namespace

std::vector<uint8_t> encode_intra_frame(const hevc_config & cfg, const yuv_image & src, yuv_image * recon)
{
	encoder e(cfg, src);
	return e.run(recon);
}

} // namespace wivrn::h267
