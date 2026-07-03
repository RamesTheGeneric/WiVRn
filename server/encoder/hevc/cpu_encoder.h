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

namespace wivrn::hevc
{

// A YUV 4:2:0 image at coded (CTB-aligned) dimensions. Luma is width x height;
// chroma planes are (width/2) x (height/2). Samples are 8-bit values in a
// 16-bit container.
struct yuv_image
{
	int width = 0;
	int height = 0;
	std::vector<uint16_t> Y;
	std::vector<uint16_t> Cb;
	std::vector<uint16_t> Cr;
};

// CPU reference intra encoder. Encodes a single all-IDR frame using 8x8 intra
// CUs (2Nx2N), DC/Planar luma modes, DCT-8 luma + DCT-4 chroma, fixed QP, no
// in-loop filters. Returns the frame's Annex-B NAL units (IDR slice only; call
// build_parameter_sets() separately for VPS/SPS/PPS). `recon`, if non-null, is
// filled with the encoder's reconstruction (which must match a conforming
// decoder's output bit-for-bit).
std::vector<uint8_t> encode_intra_frame(const hevc_config & cfg, const yuv_image & src, yuv_image * recon = nullptr);

} // namespace wivrn::hevc
