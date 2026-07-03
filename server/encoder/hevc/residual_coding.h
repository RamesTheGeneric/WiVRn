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

#include "cabac_reference.h"

#include <cstdint>

namespace wivrn::h267
{

// Encode residual_coding() (Rec. ITU-T H.265 clause 7.3.8.11) for one transform
// block of size (1<<log2size) with the given quantised coefficient levels
// (row-major [y*n + x], signed). Assumes scanIdx == 0 (up-right diagonal, which
// is what DC/Planar intra modes always use), transform_skip disabled and
// sign_data_hiding disabled. cidx: 0 = luma, 1 = Cb, 2 = Cr (chroma share the
// same contexts). The caller guarantees the block has at least one non-zero
// level (i.e. cbf == 1).
void residual_coding(cabac_encoder & cb, const int32_t * level, int log2size, int cidx);

} // namespace wivrn::h267
