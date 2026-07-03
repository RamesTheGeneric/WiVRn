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

#pragma once

// H.264 4x4 integer transform, Hadamard DC transforms, and quant/dequant
// (Rec. ITU-T H.264 clause 8.5). The INVERSE path must be byte-exact with the
// spec so our reconstruction equals a conforming decoder's output; the forward
// (encoder) quant is a standard rounding quantiser. All 4x4/2x2 arrays are
// row-major.
namespace wivrn::avc::xform
{

// Chroma QP from luma QP (Table 8-15, chroma_qp_index_offset = 0).
int chroma_qp(int qp);

// Forward 4x4 core transform (residual -> coefficients). resid/coeff: 16 ints.
void fdct4(const int * resid, int * coeff);

// Forward Hadamard transforms of the collected DC coefficients.
void fhadamard4(const int * dc, int * out); // luma 16 -> 16
void fhadamard2(const int * dc, int * out); // chroma 4 -> 4

// Forward quantisation. `intra` selects the rounding offset (1/3 vs 1/6).
int quant_ac(int coeff, int row, int col, int qp, bool intra);
int quant_dc_luma(int had, int qp, bool intra);
int quant_dc_chroma(int had, int qpc, bool intra);

// Inverse DC transforms + scaling (spec 8.5.10 / 8.5.11): produce the per-block
// scaled DC values placed at position (0,0) of each residual block.
void idc_luma(const int * level, int qp, int * dcY);    // 16 -> 16
void idc_chroma(const int * level, int qpc, int * dcC); // 4 -> 4

// Inverse 4x4 (spec 8.5.12): dequantise `level` (16, row-major) and inverse
// transform to residual `resid` (16). If use_dc, the DC (0,0) is taken from
// `dc_override` (the value from idc_luma/idc_chroma) instead of level[0].
void idct4(const int * level, int qp, bool use_dc, int dc_override, int * resid);

} // namespace wivrn::avc::xform
