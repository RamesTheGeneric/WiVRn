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

#include "cpu_encoder264.h"

#include "cavlc.h"
#include "transform264.h"

#include <array>
#include <cstring>

namespace wivrn::h264
{

using namespace xform;

namespace
{
int clip_pix(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// nC (clause 9.2.1) from a 4x4-block nnz grid: (bx,by) is the current block.
int nC_from(const std::vector<int> & nnz, int gw, int gh, int bx, int by)
{
	const bool a = bx > 0;      // left in frame
	const bool b = by > 0;      // top in frame
	const int nA = a ? nnz[(size_t)by * gw + (bx - 1)] : 0;
	const int nB = b ? nnz[(size_t)(by - 1) * gw + bx] : 0;
	if (a && b)
		return (nA + nB + 1) >> 1;
	if (a)
		return nA;
	if (b)
		return nB;
	(void)gh;
	return 0;
}

// z-scan order of the 16 luma 4x4 blocks within a macroblock, as (bx,by) 0..3.
constexpr int zblk[16][2] = {
        {0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 0}, {3, 0}, {2, 1}, {3, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 2}, {3, 2}, {2, 3}, {3, 3}};

} // namespace

std::vector<uint8_t> encode_idr_frame(const h264_config & cfg, const yuv_image & img,
                                      std::vector<uint8_t> * out_recY,
                                      std::vector<uint8_t> * out_recCb,
                                      std::vector<uint8_t> * out_recCr)
{
	const int cw = (int)cfg.coded_width(), ch = (int)cfg.coded_height();
	const int cw2 = cw / 2, ch2 = ch / 2;
	const int mbw = (int)cfg.mb_width(), mbh = (int)cfg.mb_height();
	const int qp = cfg.qp;
	const int qpc = chroma_qp(qp);

	std::vector<uint8_t> recY((size_t)cw * ch, 0), recCb((size_t)cw2 * ch2, 0), recCr((size_t)cw2 * ch2, 0);

	// 4x4-block nnz grids (TotalCoeff of the AC block) feeding nC.
	const int lgw = cw / 4, lgh = ch / 4;   // luma 4x4 grid
	const int cgw = cw2 / 4, cgh = ch2 / 4; // chroma 4x4 grid (per component)
	std::vector<int> nnzL((size_t)lgw * lgh, 0), nnzCb((size_t)cgw * cgh, 0), nnzCr((size_t)cgw * cgh, 0);

	bitwriter w;
	write_slice_header(w, cfg, 0, 0);

	for (int mby = 0; mby < mbh; ++mby)
		for (int mbx = 0; mbx < mbw; ++mbx)
		{
			const int px = mbx * 16, py = mby * 16;
			const bool availL = mbx > 0, availT = mby > 0;

			// ---- luma I_16x16 DC prediction ----
			int predY;
			{
				int sT = 0, sL = 0;
				for (int i = 0; i < 16; ++i)
				{
					if (availT)
						sT += recY[(size_t)(py - 1) * cw + (px + i)];
					if (availL)
						sL += recY[(size_t)(py + i) * cw + (px - 1)];
				}
				if (availT && availL)
					predY = (sT + sL + 16) >> 5;
				else if (availT)
					predY = (sT + 8) >> 4;
				else if (availL)
					predY = (sL + 8) >> 4;
				else
					predY = 128;
			}

			// ---- forward transform + quant of the 16 luma 4x4 blocks ----
			int dcRaw[16];                    // DC coeff per 4x4 block (raster block pos)
			std::array<std::array<int, 16>, 16> acLevel{}; // [blk raster][raster pos], pos0 unused
			bool anyLumaAC = false;
			for (int by = 0; by < 4; ++by)
				for (int bx = 0; bx < 4; ++bx)
				{
					int resid[16];
					for (int r = 0; r < 4; ++r)
						for (int c = 0; c < 4; ++c)
							resid[r * 4 + c] = (int)img.Y[(size_t)(py + by * 4 + r) * cw + (px + bx * 4 + c)] - predY;
					int coeff[16];
					fdct4(resid, coeff);
					dcRaw[by * 4 + bx] = coeff[0];
					auto & ac = acLevel[by * 4 + bx];
					for (int k = 1; k < 16; ++k)
					{
						ac[k] = quant_ac(coeff[k], k / 4, k % 4, qp, true);
						if (ac[k] != 0)
							anyLumaAC = true;
					}
				}
			// luma DC: Hadamard + quant
			int dcHad[16], dcLevel[16];
			fhadamard4(dcRaw, dcHad);
			bool anyLumaDC = false;
			for (int i = 0; i < 16; ++i)
			{
				dcLevel[i] = quant_dc_luma(dcHad[i], qp, true);
				if (dcLevel[i] != 0)
					anyLumaDC = true;
			}
			const int cbpLuma = anyLumaAC ? 15 : 0;

			// ---- luma reconstruction ----
			int dcY[16];
			idc_luma(dcLevel, qp, dcY);
			for (int by = 0; by < 4; ++by)
				for (int bx = 0; bx < 4; ++bx)
				{
					int lv[16];
					lv[0] = 0;
					for (int k = 1; k < 16; ++k)
						lv[k] = acLevel[by * 4 + bx][k];
					int resid[16];
					idct4(lv, qp, true, dcY[by * 4 + bx], resid);
					for (int r = 0; r < 4; ++r)
						for (int c = 0; c < 4; ++c)
							recY[(size_t)(py + by * 4 + r) * cw + (px + bx * 4 + c)] =
							        (uint8_t)clip_pix(predY + resid[r * 4 + c]);
				}

			// ---- chroma (Cb then Cr) ----
			int chromaDcLevel[2][4];
			int chromaPred[2][4]; // DC prediction per component/sub-block
			std::array<std::array<std::array<int, 16>, 4>, 2> chromaAC{}; // [comp][subblk][pos]
			bool anyChromaAC = false, anyChromaDC = false;
			for (int comp = 0; comp < 2; ++comp)
			{
				const uint8_t * src = comp == 0 ? img.Cb.data() : img.Cr.data();
				uint8_t * rec = comp == 0 ? recCb.data() : recCr.data();
				std::vector<int> & nnzC = comp == 0 ? nnzCb : nnzCr;
				(void)nnzC;
				const int cpx = mbx * 8, cpy = mby * 8;

				int cdcRaw[4];
				for (int sb = 0; sb < 4; ++sb)
				{
					const int sbx = (sb & 1) * 4, sby = (sb >> 1) * 4; // (0,0)(1,0)(0,1)(1,1) -> pos
					// chroma DC prediction (8.3.4): uses ONLY the MB-neighbour samples
					// (top-MB row cpy-1, left-MB column cpx-1), never intra-MB samples.
					int sT = 0, sL = 0;
					for (int i = 0; i < 4; ++i)
					{
						if (availT)
							sT += rec[(size_t)(cpy - 1) * cw2 + (cpx + sbx + i)];
						if (availL)
							sL += rec[(size_t)(cpy + sby + i) * cw2 + (cpx - 1)];
					}
					int pc;
					const bool diag = (sbx == 0 && sby == 0) || (sbx > 0 && sby > 0);
					if (diag)
					{
						if (availT && availL)
							pc = (sT + sL + 4) >> 3;
						else if (availT)
							pc = (sT + 2) >> 2;
						else if (availL)
							pc = (sL + 2) >> 2;
						else
							pc = 128;
					}
					else if (sbx > 0) // top-right: prefer top
					{
						if (availT)
							pc = (sT + 2) >> 2;
						else if (availL)
							pc = (sL + 2) >> 2;
						else
							pc = 128;
					}
					else // bottom-left: prefer left
					{
						if (availL)
							pc = (sL + 2) >> 2;
						else if (availT)
							pc = (sT + 2) >> 2;
						else
							pc = 128;
					}

					int resid[16];
					for (int r = 0; r < 4; ++r)
						for (int c = 0; c < 4; ++c)
							resid[r * 4 + c] = (int)src[(size_t)(cpy + sby + r) * cw2 + (cpx + sbx + c)] - pc;
					int coeff[16];
					fdct4(resid, coeff);
					cdcRaw[(sby / 4) * 2 + (sbx / 4)] = coeff[0];
					auto & ac = chromaAC[comp][sb];
					for (int k = 1; k < 16; ++k)
					{
						ac[k] = quant_ac(coeff[k], k / 4, k % 4, qpc, true);
						if (ac[k] != 0)
							anyChromaAC = true;
					}
					// stash prediction for reconstruction
					chromaPred[comp][sb] = pc;
				}
				int cdcHad[4];
				fhadamard2(cdcRaw, cdcHad);
				for (int i = 0; i < 4; ++i)
				{
					chromaDcLevel[comp][i] = quant_dc_chroma(cdcHad[i], qpc, true);
					if (chromaDcLevel[comp][i] != 0)
						anyChromaDC = true;
				}
			}
			const int cbpChroma = anyChromaAC ? 2 : (anyChromaDC ? 1 : 0);

			// ---- chroma reconstruction ----
			for (int comp = 0; comp < 2; ++comp)
			{
				uint8_t * rec = comp == 0 ? recCb.data() : recCr.data();
				const int cpx = mbx * 8, cpy = mby * 8;
				int dcC[4];
				idc_chroma(chromaDcLevel[comp], qpc, dcC);
				for (int sb = 0; sb < 4; ++sb)
				{
					const int sbx = (sb & 1) * 4, sby = (sb >> 1) * 4;
					int lv[16];
					lv[0] = 0;
					for (int k = 1; k < 16; ++k)
						lv[k] = chromaAC[comp][sb][k];
					int resid[16];
					idct4(lv, qpc, true, dcC[(sby / 4) * 2 + (sbx / 4)], resid);
					for (int r = 0; r < 4; ++r)
						for (int c = 0; c < 4; ++c)
							rec[(size_t)(cpy + sby + r) * cw2 + (cpx + sbx + c)] =
							        (uint8_t)clip_pix(chromaPred[comp][sb] + resid[r * 4 + c]);
				}
			}

			// ---- fill nnz grids for this MB (used by nC when coding residuals) ----
			for (int i = 0; i < 16; ++i)
			{
				const int bx = zblk[i][0], by = zblk[i][1];
				int tc = 0;
				if (cbpLuma)
					for (int k = 1; k < 16; ++k)
						tc += acLevel[by * 4 + bx][k] != 0;
				nnzL[(size_t)(mby * 4 + by) * lgw + (mbx * 4 + bx)] = tc;
			}
			for (int comp = 0; comp < 2; ++comp)
			{
				std::vector<int> & nnzC = comp == 0 ? nnzCb : nnzCr;
				for (int sb = 0; sb < 4; ++sb)
				{
					const int bx = sb & 1, by = sb >> 1;
					int tc = 0;
					if (cbpChroma & 2)
						for (int k = 1; k < 16; ++k)
							tc += chromaAC[comp][sb][k] != 0;
					nnzC[(size_t)(mby * 2 + by) * cgw + (mbx * 2 + bx)] = tc;
				}
			}

			// ---- emit MB syntax ----
			const int cbpLumaFlag = (cbpLuma == 15) ? 1 : 0;
			const int mb_type = 1 + 2 /*DC*/ + 4 * cbpChroma + 12 * cbpLumaFlag;
			w.put_ue(mb_type);
			w.put_ue(0); // intra_chroma_pred_mode = DC
			w.put_se(0); // mb_qp_delta (always coded for I16x16)

			// Intra16x16DCLevel (nC = block 0 context)
			{
				int levels[16];
				for (int s = 0; s < 16; ++s)
					levels[s] = dcLevel[zigzag4x4[s]];
				const int nc = nC_from(nnzL, lgw, lgh, mbx * 4, mby * 4);
				residual_block(w, levels, 16, nc, false);
			}
			(void)anyLumaDC;
			// Intra16x16ACLevel
			if (cbpLuma)
				for (int i = 0; i < 16; ++i)
				{
					const int bx = zblk[i][0], by = zblk[i][1];
					int levels[15];
					for (int s = 0; s < 15; ++s)
						levels[s] = acLevel[by * 4 + bx][zigzag4x4[s + 1]];
					const int nc = nC_from(nnzL, lgw, lgh, mbx * 4 + bx, mby * 4 + by);
					residual_block(w, levels, 15, nc, false);
				}
			// Chroma DC (Cb, Cr) — nC = -1
			if (cbpChroma & 3)
				for (int comp = 0; comp < 2; ++comp)
					residual_block(w, chromaDcLevel[comp], 4, -1, true);
			// Chroma AC
			if (cbpChroma & 2)
				for (int comp = 0; comp < 2; ++comp)
				{
					std::vector<int> & nnzC = comp == 0 ? nnzCb : nnzCr;
					for (int sb = 0; sb < 4; ++sb)
					{
						const int bx = sb & 1, by = sb >> 1;
						int levels[15];
						for (int s = 0; s < 15; ++s)
							levels[s] = chromaAC[comp][sb][zigzag4x4[s + 1]];
						const int nc = nC_from(nnzC, cgw, cgh, mbx * 2 + bx, mby * 2 + by);
						residual_block(w, levels, 15, nc, false);
					}
				}
		}

	w.rbsp_trailing_bits();

	std::vector<uint8_t> frame = build_parameter_sets(cfg);
	emit_nal(frame, 3, NAL_SLICE_IDR, w.bytes());

	if (out_recY)
		*out_recY = std::move(recY);
	if (out_recCb)
		*out_recCb = std::move(recCb);
	if (out_recCr)
		*out_recCr = std::move(recCr);
	return frame;
}

} // namespace wivrn::h264
