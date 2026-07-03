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

#include "param_sets.h"

namespace wivrn::h267
{

namespace
{

int ceil_log2(uint32_t v)
{
	int n = 0;
	uint32_t x = 1;
	while (x < v)
	{
		x <<= 1;
		++n;
	}
	return n;
}

// profile_tier_level(profilePresentFlag = 1, maxNumSubLayersMinus1 = 0).
// Clause 7.3.3. Produces the canonical 12-byte structure.
void write_profile_tier_level(bitwriter & w, const hevc_config & cfg)
{
	const int profile_idc = cfg.general_profile_idc();

	w.put_bits(0, 2);           // general_profile_space
	w.put_bit(0);               // general_tier_flag (Main tier)
	w.put_bits(profile_idc, 5); // general_profile_idc

	// general_profile_compatibility_flag[j] for j = 0..31. Set the bit that
	// matches general_profile_idc; a Main10 stream is also Main-compatible for
	// intra content, but we advertise only the exact profile for clarity.
	for (int j = 0; j < 32; ++j)
		w.put_bit(j == profile_idc ? 1 : 0);

	w.put_bit(1); // general_progressive_source_flag
	w.put_bit(0); // general_interlaced_source_flag
	w.put_bit(0); // general_non_packed_constraint_flag
	w.put_bit(1); // general_frame_only_constraint_flag

	// general_reserved_zero_43bits + general_inbld/reserved_zero_bit = 44 zeros.
	w.put_bits(0, 32);
	w.put_bits(0, 11);
	w.put_bit(0);

	w.put_bits(30 * 5 + 3, 8); // general_level_idc = 153 (level 5.1); adequate for VR resolutions
}

} // namespace

std::vector<uint8_t> build_vps(const hevc_config & cfg)
{
	bitwriter w;
	w.put_bits(0, 4); // vps_video_parameter_set_id
	w.put_bit(1);     // vps_base_layer_internal_flag
	w.put_bit(1);     // vps_base_layer_available_flag
	w.put_bits(0, 6); // vps_max_layers_minus1
	w.put_bits(0, 3); // vps_max_sub_layers_minus1
	w.put_bit(1);     // vps_temporal_id_nesting_flag
	w.put_bits(0xffff, 16); // vps_reserved_0xffff_16bits

	write_profile_tier_level(w, cfg);

	w.put_bit(1);  // vps_sub_layer_ordering_info_present_flag
	w.put_ue(0);   // vps_max_dec_pic_buffering_minus1[0]
	w.put_ue(0);   // vps_max_num_reorder_pics[0]
	w.put_ue(0);   // vps_max_latency_increase_plus1[0]
	w.put_bits(0, 6); // vps_max_layer_id
	w.put_ue(0);   // vps_num_layer_sets_minus1
	w.put_bit(0);  // vps_timing_info_present_flag
	w.put_bit(0);  // vps_extension_flag
	w.rbsp_trailing_bits();

	std::vector<uint8_t> nal;
	emit_nal(nal, NAL_VPS, w.bytes());
	return nal;
}

std::vector<uint8_t> build_sps(const hevc_config & cfg)
{
	bitwriter w;
	w.put_bits(0, 4); // sps_video_parameter_set_id
	w.put_bits(0, 3); // sps_max_sub_layers_minus1
	w.put_bit(1);     // sps_temporal_id_nesting_flag

	write_profile_tier_level(w, cfg);

	w.put_ue(0); // sps_seq_parameter_set_id
	w.put_ue(1); // chroma_format_idc = 1 (4:2:0)

	w.put_ue(cfg.coded_width());
	w.put_ue(cfg.coded_height());

	// Conformance window: crop the CTB-aligned coded size back to display size.
	// Offsets are in chroma sample units (SubWidthC = SubHeightC = 2 for 4:2:0).
	const uint32_t right = (cfg.coded_width() - cfg.width) / 2;
	const uint32_t bottom = (cfg.coded_height() - cfg.height) / 2;
	if (right || bottom)
	{
		w.put_bit(1); // conformance_window_flag
		w.put_ue(0);  // conf_win_left_offset
		w.put_ue(right);
		w.put_ue(0);  // conf_win_top_offset
		w.put_ue(bottom);
	}
	else
	{
		w.put_bit(0);
	}

	w.put_ue(cfg.bit_depth - 8); // bit_depth_luma_minus8
	w.put_ue(cfg.bit_depth - 8); // bit_depth_chroma_minus8
	w.put_ue(4);                 // log2_max_pic_order_cnt_lsb_minus4 (16-bit POC lsb)

	w.put_bit(1); // sps_sub_layer_ordering_info_present_flag
	w.put_ue(0);  // sps_max_dec_pic_buffering_minus1[0]  (1 picture: current only)
	w.put_ue(0);  // sps_max_num_reorder_pics[0]
	w.put_ue(0);  // sps_max_latency_increase_plus1[0]

	w.put_ue(cfg.min_cb_log2_size - 3);                     // log2_min_luma_coding_block_size_minus3
	w.put_ue(cfg.ctb_log2_size - cfg.min_cb_log2_size);     // log2_diff_max_min_luma_coding_block_size
	w.put_ue(cfg.min_tb_log2_size - 2);                     // log2_min_luma_transform_block_size_minus2
	w.put_ue(cfg.max_tb_log2_size - cfg.min_tb_log2_size);  // log2_diff_max_min_luma_transform_block_size
	w.put_ue(0); // max_transform_hierarchy_depth_inter
	w.put_ue(0); // max_transform_hierarchy_depth_intra

	w.put_bit(0); // scaling_list_enabled_flag
	w.put_bit(0); // amp_enabled_flag
	w.put_bit(0); // sample_adaptive_offset_enabled_flag
	w.put_bit(0); // pcm_enabled_flag
	w.put_ue(0);  // num_short_term_ref_pic_sets
	w.put_bit(0); // long_term_ref_pics_present_flag
	w.put_bit(0); // sps_temporal_mvp_enabled_flag
	w.put_bit(0); // strong_intra_smoothing_enabled_flag
	w.put_bit(0); // vui_parameters_present_flag
	w.put_bit(0); // sps_extension_present_flag
	w.rbsp_trailing_bits();

	std::vector<uint8_t> nal;
	emit_nal(nal, NAL_SPS, w.bytes());
	return nal;
}

std::vector<uint8_t> build_pps(const hevc_config & cfg)
{
	bitwriter w;
	w.put_ue(0); // pps_pic_parameter_set_id
	w.put_ue(0); // pps_seq_parameter_set_id
	w.put_bit(0); // dependent_slice_segments_enabled_flag
	w.put_bit(0); // output_flag_present_flag
	w.put_bits(0, 3); // num_extra_slice_header_bits
	w.put_bit(0); // sign_data_hiding_enabled_flag
	w.put_bit(0); // cabac_init_present_flag
	w.put_ue(0);  // num_ref_idx_l0_default_active_minus1
	w.put_ue(0);  // num_ref_idx_l1_default_active_minus1
	w.put_se(cfg.qp - 26); // init_qp_minus26
	w.put_bit(0); // constrained_intra_pred_flag
	w.put_bit(0); // transform_skip_enabled_flag
	w.put_bit(0); // cu_qp_delta_enabled_flag
	w.put_se(0);  // pps_cb_qp_offset
	w.put_se(0);  // pps_cr_qp_offset
	w.put_bit(0); // pps_slice_chroma_qp_offsets_present_flag
	w.put_bit(0); // weighted_pred_flag
	w.put_bit(0); // weighted_bipred_flag
	w.put_bit(0); // transquant_bypass_enabled_flag
	w.put_bit(0); // tiles_enabled_flag
	w.put_bit(0); // entropy_coding_sync_enabled_flag (WPP off for MVP)
	w.put_bit(0); // pps_loop_filter_across_slices_enabled_flag

	w.put_bit(1); // deblocking_filter_control_present_flag
	w.put_bit(0); //   deblocking_filter_override_enabled_flag
	w.put_bit(1); //   pps_deblocking_filter_disabled_flag (deblocking OFF)

	w.put_bit(0); // pps_scaling_list_data_present_flag
	w.put_bit(0); // lists_modification_present_flag
	w.put_ue(0);  // log2_parallel_merge_level_minus2
	w.put_bit(0); // slice_segment_header_extension_present_flag
	w.put_bit(0); // pps_extension_present_flag
	w.rbsp_trailing_bits();

	std::vector<uint8_t> nal;
	emit_nal(nal, NAL_PPS, w.bytes());
	return nal;
}

std::vector<uint8_t> build_parameter_sets(const hevc_config & cfg)
{
	std::vector<uint8_t> out;
	auto append = [&out](const std::vector<uint8_t> & nal) { out.insert(out.end(), nal.begin(), nal.end()); };
	append(build_vps(cfg));
	append(build_sps(cfg));
	append(build_pps(cfg));
	return out;
}

void write_slice_header(bitwriter & w, const hevc_config & cfg, uint32_t slice_ctb_addr, bool first_in_pic)
{
	w.put_bit(first_in_pic ? 1 : 0); // first_slice_segment_in_pic_flag
	// nal_unit_type is an IDR (in [BLA_W_LP=16 .. RSV_IRAP_VCL23=23]):
	w.put_bit(0); // no_output_of_prior_pics_flag
	w.put_ue(0);  // slice_pic_parameter_set_id

	if (not first_in_pic)
	{
		// dependent_slice_segments_enabled_flag is 0, so no dependent flag.
		const int bits = ceil_log2(cfg.ctbs_total());
		w.put_bits(slice_ctb_addr, bits); // slice_segment_address
	}

	// num_extra_slice_header_bits == 0 -> nothing.
	w.put_ue(2); // slice_type = I

	// nal_unit_type is IDR -> skip pic_order_cnt / ref pic set / temporal mvp.
	// sample_adaptive_offset_enabled_flag == 0 -> no slice SAO flags.
	// I-slice -> no ref list / mvd / cabac_init / collocated / five_minus_max_merge.

	w.put_se(0); // slice_qp_delta (SliceQpY == init_qp)

	// pps_slice_chroma_qp_offsets_present_flag == 0.
	// deblocking_filter_override_enabled_flag == 0.
	// deblocking disabled in PPS + SAO off + loop_filter_across_slices off in PPS
	//   -> no slice_loop_filter_across_slices_enabled_flag.
	// tiles and entropy_coding_sync disabled -> no entry point offsets.
	// slice_segment_header_extension_present_flag == 0.

	w.byte_alignment(); // align so CABAC slice data starts on a byte boundary
}

} // namespace wivrn::h267
