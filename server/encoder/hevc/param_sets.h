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

#include "bitwriter.h"

#include <cstdint>
#include <vector>

namespace wivrn::h267
{

// HEVC NAL unit types used by this encoder (Rec. ITU-T H.265 Table 7-1).
enum nal_type : int
{
	NAL_TRAIL_R = 1,
	NAL_IDR_W_RADL = 19,
	NAL_IDR_N_LP = 20,
	NAL_VPS = 32,
	NAL_SPS = 33,
	NAL_PPS = 34,
};

// Fixed configuration for the intra-only, filters-off MVP encoder. The coded
// picture is padded up to a whole number of CTBs so every CTB is complete
// (simplifies the coding tree); the conformance window crops back to the real
// display size.
struct hevc_config
{
	uint32_t width = 0;  // real (display) luma width, must be even
	uint32_t height = 0; // real (display) luma height, must be even
	int bit_depth = 8;   // 8 (Main) or 10 (Main10)
	int qp = 26;         // slice/init QP

	int ctb_log2_size = 6;   // 64x64 coding tree block
	int min_cb_log2_size = 3; // 8x8 minimum coding block
	int min_tb_log2_size = 2; // 4x4 minimum transform block
	int max_tb_log2_size = 5; // 32x32 maximum transform block

	uint32_t num_slices = 1; // independent slices per picture (>=1)

	// Testing/bring-up: force every luma CU to this intra mode (0=Planar, 1=DC).
	// -1 = normal SAD-based DC/Planar decision. Used to compare against the GPU
	// wavefront which currently implements DC only.
	int force_luma_mode = -1;

	// Derived helpers.
	int general_profile_idc() const { return bit_depth > 8 ? 2 : 1; } // Main10 : Main
	uint32_t ctb_size() const { return 1u << ctb_log2_size; }
	uint32_t coded_width() const { return ((width + ctb_size() - 1) / ctb_size()) * ctb_size(); }
	uint32_t coded_height() const { return ((height + ctb_size() - 1) / ctb_size()) * ctb_size(); }
	uint32_t ctbs_x() const { return coded_width() / ctb_size(); }
	uint32_t ctbs_y() const { return coded_height() / ctb_size(); }
	uint32_t ctbs_total() const { return ctbs_x() * ctbs_y(); }
};

// Build complete Annex-B NAL units (start code + header + emulation-prevented
// RBSP) for the video/sequence/picture parameter sets. Each returned vector is
// one NAL unit ready to hand to the network layer.
std::vector<uint8_t> build_vps(const hevc_config & cfg);
std::vector<uint8_t> build_sps(const hevc_config & cfg);
std::vector<uint8_t> build_pps(const hevc_config & cfg);

// Convenience: VPS ++ SPS ++ PPS concatenated (the parameter-set bundle sent on
// every IDR over the reliable channel).
std::vector<uint8_t> build_parameter_sets(const hevc_config & cfg);

// Write the slice_segment_header() for an IDR I-slice into `w`, ending with
// byte_alignment() so the CABAC-coded slice data can follow byte-aligned.
// `slice_ctb_addr` is the raster CTB address of the slice's first CTB (0 for a
// single-slice picture); `first_in_pic` marks the first slice segment.
void write_slice_header(bitwriter & w, const hevc_config & cfg, uint32_t slice_ctb_addr, bool first_in_pic);

} // namespace wivrn::h267
