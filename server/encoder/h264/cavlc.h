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

#include "param_sets264.h"

namespace wivrn::h264
{

// CAVLC residual_block (Rec. ITU-T H.264 clause 9.2). `levels` holds the block's
// coefficients in scan order (index 0 = lowest frequency), `count` = maxNumCoeff
// (16 luma DC, 15 AC, 4 chroma DC). `nC` selects the coeff_token table (use -1
// for chroma DC). Returns the number of non-zero coefficients (TotalCoeff) so
// the caller can maintain the neighbour nnz map that feeds `nC`.
int residual_block(bitwriter & w, const int * levels, int count, int nC, bool chroma_dc);

// The 4x4 up-right zig-zag scan (raster index for each scan position).
extern const int zigzag4x4[16];

} // namespace wivrn::h264
