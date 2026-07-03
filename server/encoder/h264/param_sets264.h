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

#include "../hevc/bitwriter.h"

#include <cstdint>
#include <vector>

// H.264 / AVC parameter sets and framing for the shader-based "h2-67" encoder's
// H.264 path. Constrained Baseline profile, all-IDR I-slices, 4:2:0 8-bit,
// deblocking off, fixed QP — the CAVLC counterpart of the HEVC MVP.
namespace wivrn::avc
{

// Reuse the codec-agnostic RBSP bit writer + emulation prevention from the HEVC
// side (MSB-first packing, Exp-Golomb, EBSP are identical between H.264/H.265).
using bitwriter = wivrn::h267::bitwriter;

enum nal_type : int
{
	NAL_SLICE_NON_IDR = 1,
	NAL_SLICE_IDR = 5,
	NAL_SPS = 7,
	NAL_PPS = 8,
};

struct h264_config
{
	uint16_t width = 0;
	uint16_t height = 0;
	int bit_depth = 8; // 8-bit only
	int qp = 26;       // fixed QP (0..51)

	uint32_t mb_width() const { return (width + 15u) / 16u; }
	uint32_t mb_height() const { return (height + 15u) / 16u; }
	uint32_t coded_width() const { return mb_width() * 16u; }
	uint32_t coded_height() const { return mb_height() * 16u; }
	uint32_t mbs_total() const { return mb_width() * mb_height(); }
};

// Emit a complete Annex-B H.264 NAL: 00 00 00 01 start code, the 1-byte NAL
// header (forbidden_zero_bit=0 | nal_ref_idc | nal_unit_type), then the
// emulation-prevented RBSP. `rbsp` must already end with rbsp_trailing_bits.
void emit_nal(std::vector<uint8_t> & out, int nal_ref_idc, int nal_unit_type,
              const std::vector<uint8_t> & rbsp);

// Build SPS + PPS NALs (concatenated Annex-B) for this config. Sent once per IDR.
std::vector<uint8_t> build_parameter_sets(const h264_config & cfg);

// Write the IDR I-slice header (first_mb_in_slice..slice_qp_delta + deblocking
// idc). CAVLC: the slice data follows immediately, NOT byte-aligned.
void write_slice_header(bitwriter & w, const h264_config & cfg, uint32_t first_mb, uint16_t idr_pic_id);

} // namespace wivrn::avc
