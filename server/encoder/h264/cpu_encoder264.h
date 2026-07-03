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

#include <cstdint>
#include <vector>

namespace wivrn::avc
{

// Coded-size (CTB/MB-aligned) 8-bit YUV 4:2:0 planes. Chroma is half resolution.
struct yuv_image
{
	int width = 0;  // = coded_width
	int height = 0; // = coded_height
	std::vector<uint8_t> Y;
	std::vector<uint8_t> Cb;
	std::vector<uint8_t> Cr;
};

// Encode one all-IDR I-slice frame: I_16x16 DC intra, single slice, fixed QP,
// deblocking off. Returns the complete Annex-B frame (SPS + PPS + IDR slice).
// If the recon pointers are non-null they receive the coded-size reconstruction
// (what a conforming decoder produces) for validation.
std::vector<uint8_t> encode_idr_frame(const h264_config & cfg, const yuv_image & img,
                                      std::vector<uint8_t> * recY = nullptr,
                                      std::vector<uint8_t> * recCb = nullptr,
                                      std::vector<uint8_t> * recCr = nullptr);

} // namespace wivrn::avc
