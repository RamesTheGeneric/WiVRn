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

#include "param_sets264.h"

namespace wivrn::h264
{

void emit_nal(std::vector<uint8_t> & out, int nal_ref_idc, int nal_unit_type,
              const std::vector<uint8_t> & rbsp)
{
	out.push_back(0);
	out.push_back(0);
	out.push_back(0);
	out.push_back(1);
	out.push_back((uint8_t)(((nal_ref_idc & 3) << 5) | (nal_unit_type & 0x1f)));
	wivrn::h267::append_ebsp(out, rbsp);
}

namespace
{
// Rec. ITU-T H.264 7.3.2.1.1 seq_parameter_set_rbsp (Constrained Baseline).
std::vector<uint8_t> build_sps(const h264_config & cfg)
{
	bitwriter w;
	w.put_bits(66, 8);  // profile_idc = Baseline
	w.put_bit(1);       // constraint_set0_flag
	w.put_bit(1);       // constraint_set1_flag (=> Constrained Baseline)
	w.put_bit(0);       // constraint_set2_flag
	w.put_bit(0);       // constraint_set3_flag
	w.put_bit(0);       // constraint_set4_flag
	w.put_bit(0);       // constraint_set5_flag
	w.put_bits(0, 2);   // reserved_zero_2bits
	w.put_bits(51, 8);  // level_idc = 5.1 (avoid level constraints at VR res)
	w.put_ue(0);        // seq_parameter_set_id
	// profile_idc 66 => no chroma_format_idc / bit-depth syntax (4:2:0 8-bit implied)
	w.put_ue(0);        // log2_max_frame_num_minus4 (frame_num is 4 bits)
	w.put_ue(2);        // pic_order_cnt_type = 2 (POC derived, no slice POC syntax)
	w.put_ue(1);        // max_num_ref_frames
	w.put_bit(0);       // gaps_in_frame_num_value_allowed_flag
	w.put_ue(cfg.mb_width() - 1);  // pic_width_in_mbs_minus1
	w.put_ue(cfg.mb_height() - 1); // pic_height_in_map_units_minus1
	w.put_bit(1);       // frame_mbs_only_flag
	w.put_bit(1);       // direct_8x8_inference_flag

	// frame_cropping: crop the CTB/MB padding back to the display size. 4:2:0 =>
	// CropUnitX = SubWidthC = 2, CropUnitY = SubHeightC * (2 - frame_mbs_only) = 2.
	const uint32_t crop_r = (cfg.coded_width() - cfg.width) / 2;
	const uint32_t crop_b = (cfg.coded_height() - cfg.height) / 2;
	if (crop_r || crop_b)
	{
		w.put_bit(1);       // frame_cropping_flag
		w.put_ue(0);        // frame_crop_left_offset
		w.put_ue(crop_r);   // frame_crop_right_offset
		w.put_ue(0);        // frame_crop_top_offset
		w.put_ue(crop_b);   // frame_crop_bottom_offset
	}
	else
	{
		w.put_bit(0); // frame_cropping_flag
	}
	w.put_bit(0); // vui_parameters_present_flag
	w.rbsp_trailing_bits();
	return w.bytes();
}

// Rec. ITU-T H.264 7.3.2.2 pic_parameter_set_rbsp.
std::vector<uint8_t> build_pps(const h264_config & cfg)
{
	bitwriter w;
	w.put_ue(0);  // pic_parameter_set_id
	w.put_ue(0);  // seq_parameter_set_id
	w.put_bit(0); // entropy_coding_mode_flag = 0 (CAVLC)
	w.put_bit(0); // bottom_field_pic_order_in_frame_present_flag
	w.put_ue(0);  // num_slice_groups_minus1
	w.put_ue(0);  // num_ref_idx_l0_default_active_minus1
	w.put_ue(0);  // num_ref_idx_l1_default_active_minus1
	w.put_bit(0); // weighted_pred_flag
	w.put_bits(0, 2); // weighted_bipred_idc
	w.put_se(cfg.qp - 26); // pic_init_qp_minus26
	w.put_se(0);  // pic_init_qs_minus26
	w.put_se(0);  // chroma_qp_index_offset
	w.put_bit(1); // deblocking_filter_control_present_flag (to disable in slice hdr)
	w.put_bit(0); // constrained_intra_pred_flag
	w.put_bit(0); // redundant_pic_cnt_present_flag
	w.rbsp_trailing_bits();
	return w.bytes();
}
} // namespace

std::vector<uint8_t> build_parameter_sets(const h264_config & cfg)
{
	std::vector<uint8_t> out;
	emit_nal(out, 3, NAL_SPS, build_sps(cfg));
	emit_nal(out, 3, NAL_PPS, build_pps(cfg));
	return out;
}

void write_slice_header(bitwriter & w, const h264_config & cfg, uint32_t first_mb, uint16_t idr_pic_id)
{
	(void)cfg;
	w.put_ue(first_mb); // first_mb_in_slice
	w.put_ue(7);        // slice_type = 7 (I, all slices in pic are I)
	w.put_ue(0);        // pic_parameter_set_id
	w.put_bits(0, 4);   // frame_num (log2_max_frame_num = 4 bits) = 0
	// frame_mbs_only_flag = 1 => no field_pic_flag
	w.put_ue(idr_pic_id); // idr_pic_id (NAL type 5)
	// pic_order_cnt_type == 2 => no pic_order_cnt_lsb
	// dec_ref_pic_marking (IDR, nal_ref_idc != 0):
	w.put_bit(0); // no_output_of_prior_pics_flag
	w.put_bit(0); // long_term_reference_flag
	// entropy_coding_mode_flag == 0 => no cabac_init_idc
	w.put_se(0);  // slice_qp_delta (SliceQP == pic_init_qp)
	// deblocking_filter_control_present_flag == 1:
	w.put_ue(1);  // disable_deblocking_filter_idc = 1 (off; no alpha/beta offsets)
	// num_slice_groups_minus1 == 0 => no slice_group_change_cycle
	// CAVLC: slice_data() follows immediately, no byte alignment.
}

} // namespace wivrn::h264
