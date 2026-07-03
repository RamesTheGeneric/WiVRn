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

// Constant tables from Rec. ITU-T H.265 clause 9.3 (CABAC) and the integer
// transforms of clause 8.6. The context initialisation values and their layout
// match the reference decoders (FFmpeg / libde265); any deviation here would
// desynchronise a conforming decoder, so these must stay byte-exact.
namespace wivrn::hevc::tables
{

// "Context Not Used" default initialisation value.
inline constexpr uint8_t CNU = 154;

// Byte offset of each syntax element's context block within the context array.
// Order and sizes match FFmpeg's libavcodec/hevc/cabac.c so the init_values
// table below can be indexed directly.
enum ctx_offset : int
{
	SAO_MERGE_FLAG = 0,                              // 1
	SAO_TYPE_IDX = SAO_MERGE_FLAG + 1,              // 1
	SPLIT_CODING_UNIT_FLAG = SAO_TYPE_IDX + 1,      // 3
	CU_TRANSQUANT_BYPASS_FLAG = SPLIT_CODING_UNIT_FLAG + 3, // 1
	SKIP_FLAG = CU_TRANSQUANT_BYPASS_FLAG + 1,      // 3
	CU_QP_DELTA = SKIP_FLAG + 3,                    // 3
	PRED_MODE_FLAG = CU_QP_DELTA + 3,               // 1
	PART_MODE = PRED_MODE_FLAG + 1,                 // 4
	PREV_INTRA_LUMA_PRED_FLAG = PART_MODE + 4,      // 1
	INTRA_CHROMA_PRED_MODE = PREV_INTRA_LUMA_PRED_FLAG + 1, // 2
	MERGE_FLAG = INTRA_CHROMA_PRED_MODE + 2,        // 1
	MERGE_IDX = MERGE_FLAG + 1,                     // 1
	INTER_PRED_IDC = MERGE_IDX + 1,                 // 5
	REF_IDX_L0 = INTER_PRED_IDC + 5,               // 2
	REF_IDX_L1 = REF_IDX_L0 + 2,                    // 2
	ABS_MVD_GREATER0_FLAG = REF_IDX_L1 + 2,         // 2
	ABS_MVD_GREATER1_FLAG = ABS_MVD_GREATER0_FLAG + 2, // 2
	MVP_LX_FLAG = ABS_MVD_GREATER1_FLAG + 2,        // 1
	NO_RESIDUAL_DATA_FLAG = MVP_LX_FLAG + 1,        // 1
	SPLIT_TRANSFORM_FLAG = NO_RESIDUAL_DATA_FLAG + 1, // 3
	CBF_LUMA = SPLIT_TRANSFORM_FLAG + 3,            // 2
	CBF_CB_CR = CBF_LUMA + 2,                       // 5
	TRANSFORM_SKIP_FLAG = CBF_CB_CR + 5,            // 2
	EXPLICIT_RDPCM_FLAG = TRANSFORM_SKIP_FLAG + 2,  // 2
	EXPLICIT_RDPCM_DIR_FLAG = EXPLICIT_RDPCM_FLAG + 2, // 2
	LAST_SIGNIFICANT_COEFF_X_PREFIX = EXPLICIT_RDPCM_DIR_FLAG + 2, // 18
	LAST_SIGNIFICANT_COEFF_Y_PREFIX = LAST_SIGNIFICANT_COEFF_X_PREFIX + 18, // 18
	SIGNIFICANT_COEFF_GROUP_FLAG = LAST_SIGNIFICANT_COEFF_Y_PREFIX + 18, // 4
	SIGNIFICANT_COEFF_FLAG = SIGNIFICANT_COEFF_GROUP_FLAG + 4, // 44
	COEFF_ABS_LEVEL_GREATER1_FLAG = SIGNIFICANT_COEFF_FLAG + 44, // 24
	COEFF_ABS_LEVEL_GREATER2_FLAG = COEFF_ABS_LEVEL_GREATER1_FLAG + 24, // 6
	LOG2_RES_SCALE_ABS = COEFF_ABS_LEVEL_GREATER2_FLAG + 6, // 8
	RES_SCALE_SIGN_FLAG = LOG2_RES_SCALE_ABS + 8,   // 2
	CU_CHROMA_QP_OFFSET_FLAG = RES_SCALE_SIGN_FLAG + 2, // 1
	CU_CHROMA_QP_OFFSET_IDX = CU_CHROMA_QP_OFFSET_FLAG + 1, // 1
	HEVC_CONTEXTS = CU_CHROMA_QP_OFFSET_IDX + 1,    // = 179
};

// init_values[initType][ctxIdx] (Rec. ITU-T H.265 Tables 9-5..9-37).
// initType 0 = I slices; 1,2 = P/B. Verbatim from the reference decoders.
inline constexpr uint8_t init_values[3][HEVC_CONTEXTS] = {
        {
                // sao_merge_flag
                153,
                // sao_type_idx
                200,
                // split_coding_unit_flag
                139, 141, 157,
                // cu_transquant_bypass_flag
                154,
                // skip_flag
                CNU, CNU, CNU,
                // cu_qp_delta
                154, 154, 154,
                // pred_mode
                CNU,
                // part_mode
                184, CNU, CNU, CNU,
                // prev_intra_luma_pred_flag
                184,
                // intra_chroma_pred_mode
                63, 139,
                // merge_flag
                CNU,
                // merge_idx
                CNU,
                // inter_pred_idc
                CNU, CNU, CNU, CNU, CNU,
                // ref_idx_l0
                CNU, CNU,
                // ref_idx_l1
                CNU, CNU,
                // abs_mvd_greater0_flag
                CNU, CNU,
                // abs_mvd_greater1_flag
                CNU, CNU,
                // mvp_lx_flag
                CNU,
                // no_residual_data_flag
                CNU,
                // split_transform_flag
                153, 138, 138,
                // cbf_luma
                111, 141,
                // cbf_cb, cbf_cr
                94, 138, 182, 154, 154,
                // transform_skip_flag
                139, 139,
                // explicit_rdpcm_flag
                139, 139,
                // explicit_rdpcm_dir_flag
                139, 139,
                // last_significant_coeff_x_prefix
                110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,
                79, 108, 123, 63,
                // last_significant_coeff_y_prefix
                110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111,
                79, 108, 123, 63,
                // significant_coeff_group_flag
                91, 171, 134, 141,
                // significant_coeff_flag
                111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153,
                125, 107, 125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140,
                139, 182, 182, 152, 136, 152, 136, 153, 136, 139, 111, 136, 139, 111,
                141, 111,
                // coeff_abs_level_greater1_flag
                140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107,
                122, 152, 140, 179, 166, 182, 140, 227, 122, 197,
                // coeff_abs_level_greater2_flag
                138, 153, 136, 167, 152, 152,
                // log2_res_scale_abs
                154, 154, 154, 154, 154, 154, 154, 154,
                // res_scale_sign_flag
                154, 154,
                // cu_chroma_qp_offset_flag
                154,
                // cu_chroma_qp_offset_idx
                154,
        },
        {
                // sao_merge_flag
                153,
                // sao_type_idx
                185,
                // split_coding_unit_flag
                107, 139, 126,
                // cu_transquant_bypass_flag
                154,
                // skip_flag
                197, 185, 201,
                // cu_qp_delta
                154, 154, 154,
                // pred_mode
                149,
                // part_mode
                154, 139, 154, 154,
                // prev_intra_luma_pred_flag
                154,
                // intra_chroma_pred_mode
                152, 139,
                // merge_flag
                110,
                // merge_idx
                122,
                // inter_pred_idc
                95, 79, 63, 31, 31,
                // ref_idx_l0
                153, 153,
                // ref_idx_l1
                153, 153,
                // abs_mvd_greater0_flag
                140, 198,
                // abs_mvd_greater1_flag
                140, 198,
                // mvp_lx_flag
                168,
                // no_residual_data_flag
                79,
                // split_transform_flag
                124, 138, 94,
                // cbf_luma
                153, 111,
                // cbf_cb, cbf_cr
                149, 107, 167, 154, 154,
                // transform_skip_flag
                139, 139,
                // explicit_rdpcm_flag
                139, 139,
                // explicit_rdpcm_dir_flag
                139, 139,
                // last_significant_coeff_x_prefix
                125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95,
                94, 108, 123, 108,
                // last_significant_coeff_y_prefix
                125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95,
                94, 108, 123, 108,
                // significant_coeff_group_flag
                121, 140, 61, 154,
                // significant_coeff_flag
                155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153,
                154, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170,
                153, 123, 123, 107, 121, 107, 121, 167, 151, 183, 140, 151, 183, 140,
                140, 140,
                // coeff_abs_level_greater1_flag
                154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121,
                136, 137, 169, 194, 166, 167, 154, 167, 137, 182,
                // coeff_abs_level_greater2_flag
                107, 167, 91, 122, 107, 167,
                // log2_res_scale_abs
                154, 154, 154, 154, 154, 154, 154, 154,
                // res_scale_sign_flag
                154, 154,
                // cu_chroma_qp_offset_flag
                154,
                // cu_chroma_qp_offset_idx
                154,
        },
        {
                // sao_merge_flag
                153,
                // sao_type_idx
                160,
                // split_coding_unit_flag
                107, 139, 126,
                // cu_transquant_bypass_flag
                154,
                // skip_flag
                197, 185, 201,
                // cu_qp_delta
                154, 154, 154,
                // pred_mode
                134,
                // part_mode
                154, 139, 154, 154,
                // prev_intra_luma_pred_flag
                183,
                // intra_chroma_pred_mode
                152, 139,
                // merge_flag
                154,
                // merge_idx
                137,
                // inter_pred_idc
                95, 79, 63, 31, 31,
                // ref_idx_l0
                153, 153,
                // ref_idx_l1
                153, 153,
                // abs_mvd_greater0_flag
                169, 198,
                // abs_mvd_greater1_flag
                169, 198,
                // mvp_lx_flag
                168,
                // no_residual_data_flag
                79,
                // split_transform_flag
                224, 167, 122,
                // cbf_luma
                153, 111,
                // cbf_cb, cbf_cr
                149, 92, 167, 154, 154,
                // transform_skip_flag
                139, 139,
                // explicit_rdpcm_flag
                139, 139,
                // explicit_rdpcm_dir_flag
                139, 139,
                // last_significant_coeff_x_prefix
                125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111,
                79, 108, 123, 93,
                // last_significant_coeff_y_prefix
                125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111,
                79, 108, 123, 93,
                // significant_coeff_group_flag
                121, 140, 61, 154,
                // significant_coeff_flag
                170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153,
                154, 166, 183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170,
                153, 138, 138, 122, 121, 122, 121, 167, 151, 183, 140, 151, 183, 140,
                140, 140,
                // coeff_abs_level_greater1_flag
                154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121,
                136, 122, 169, 208, 166, 167, 154, 152, 167, 182,
                // coeff_abs_level_greater2_flag
                107, 167, 91, 107, 107, 167,
                // log2_res_scale_abs
                154, 154, 154, 154, 154, 154, 154, 154,
                // res_scale_sign_flag
                154, 154,
                // cu_chroma_qp_offset_flag
                154,
                // cu_chroma_qp_offset_idx
                154,
        },
};

// rangeTabLps[pStateIdx][qRangeIdx] (Rec. ITU-T H.265 Table 9-46).
inline constexpr uint8_t rangeTabLps[64][4] = {
        {128, 176, 208, 240}, {128, 167, 197, 227}, {128, 158, 187, 216}, {123, 150, 178, 205},
        {116, 142, 169, 195}, {111, 135, 160, 185}, {105, 128, 152, 175}, {100, 122, 144, 166},
        {95, 116, 137, 158}, {90, 110, 130, 150}, {85, 104, 123, 142}, {81, 99, 117, 135},
        {77, 94, 111, 128}, {73, 89, 105, 122}, {69, 85, 100, 116}, {66, 80, 95, 110},
        {62, 76, 90, 104}, {59, 72, 86, 99}, {56, 69, 81, 94}, {53, 65, 77, 89},
        {51, 62, 73, 85}, {48, 59, 69, 80}, {46, 56, 66, 76}, {43, 53, 63, 72},
        {41, 50, 59, 69}, {39, 48, 56, 65}, {37, 45, 54, 62}, {35, 43, 51, 59},
        {33, 41, 48, 56}, {32, 39, 46, 53}, {30, 37, 43, 50}, {29, 35, 41, 48},
        {27, 33, 39, 45}, {26, 31, 37, 43}, {24, 30, 35, 41}, {23, 28, 33, 39},
        {22, 27, 32, 37}, {21, 26, 30, 35}, {20, 24, 29, 33}, {19, 23, 27, 31},
        {18, 22, 26, 30}, {17, 21, 25, 28}, {16, 20, 23, 27}, {15, 19, 22, 25},
        {14, 18, 21, 24}, {14, 17, 20, 23}, {13, 16, 19, 22}, {12, 15, 18, 21},
        {12, 14, 17, 20}, {11, 14, 16, 19}, {11, 13, 15, 18}, {10, 12, 15, 17},
        {10, 12, 14, 16}, {9, 11, 13, 15}, {9, 11, 12, 14}, {8, 10, 12, 14},
        {8, 9, 11, 13}, {7, 9, 11, 12}, {7, 9, 10, 12}, {7, 8, 10, 11},
        {6, 8, 9, 11}, {6, 7, 9, 10}, {6, 7, 8, 9}, {2, 2, 2, 2},
};

// transIdxLps / transIdxMps (Rec. ITU-T H.265 Table 9-47).
inline constexpr uint8_t transIdxLps[64] = {
        0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
        13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
        24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
        33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
};
inline constexpr uint8_t transIdxMps[64] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
};

} // namespace wivrn::hevc::tables
