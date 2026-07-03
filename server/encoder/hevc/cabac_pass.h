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

#include "param_sets.h"

#include <cstdint>
#include <vector>

namespace wivrn::h267
{

// Precomputed per-8x8-CU coding data produced by the reconstruction pass (the
// GPU wavefront), laid out in raster CU order: index = (y0/8) * (coded_width/8)
// + (x0/8). This is the seam between the parallel reconstruction stage and the
// serial entropy-coding stage.
struct block_syntax
{
	std::vector<uint8_t> mode;     // luma intra mode per CU (0=Planar, 1=DC)
	std::vector<uint8_t> cbf_luma; // per CU
	std::vector<uint8_t> cbf_cb;
	std::vector<uint8_t> cbf_cr;
	std::vector<int32_t> lev_y;  // 64 per CU (8x8)
	std::vector<int32_t> lev_cb; // 16 per CU (4x4)
	std::vector<int32_t> lev_cr;
};

// Entropy-code a whole IDR I-slice from the precomputed block data, producing
// the Annex-B slice NAL. This is the CPU reference for the CABAC stage: it
// performs no reconstruction, only syntax generation, and reuses the verified
// residual_coding / CABAC engine. (Call build_parameter_sets() separately.)
std::vector<uint8_t> encode_slice_from_syntax(const hevc_config & cfg, const block_syntax & bs);

} // namespace wivrn::h267
