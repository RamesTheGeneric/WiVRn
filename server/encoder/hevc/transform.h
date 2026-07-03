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

#include <cstdint>

// HEVC integer transforms and quantisation (Rec. ITU-T H.265 clauses 8.6.3 and
// 8.6.4). The inverse transform and inverse quantisation are byte-exact to the
// spec so that the encoder's reconstruction matches any conforming decoder; the
// forward transform / quantisation are the encoder's own (quality) choice.
//
// This reference supports 4x4 and 8x8 DCT-II (the sizes used by the first
// real-residual encoder: 8x8 luma, 4x4 chroma). All arrays are row-major [y][x].
namespace wivrn::h267::xform
{

// Forward DCT-II of an nxn residual block (n = 4 or 8). in: residual samples
// (already prediction-subtracted), out: transform coefficients.
void fdct(const int32_t * in, int32_t * out, int n, int bit_depth);

// Inverse DCT-II (spec 8.6.4). in: dequantised coefficients, out: residual.
void idct(const int32_t * in, int32_t * out, int n, int bit_depth);

// Forward quantisation (encoder choice): coeff -> level, with an intra dead-zone
// rounding offset. Returns the level array (signed).
void quant(const int32_t * coeff, int32_t * level, int n, int qp, int bit_depth);

// Inverse quantisation (spec 8.6.3): level -> dequantised coefficient.
void dequant(const int32_t * level, int32_t * coeff, int n, int qp, int bit_depth);

} // namespace wivrn::h267::xform
