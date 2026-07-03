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

#include "residual_coding.h"

#include <array>
#include <cstdlib>

namespace wivrn::hevc
{

using namespace tables;

namespace
{

struct pos
{
	int x, y;
};

// Up-right diagonal scan (Rec. ITU-T H.265 clause 6.5.3) for a size x size grid.
void diag_scan(int size, pos * out)
{
	int i = 0, x = 0, y = 0;
	bool stop = false;
	while (not stop)
	{
		while (y >= 0)
		{
			if (x < size && y < size)
				out[i++] = {x, y};
			--y;
			++x;
		}
		y = x;
		x = 0;
		if (i >= size * size)
			stop = true;
	}
}

// Group index / minimum-in-group tables for last_significant_coeff (9.3.3.3).
constexpr int group_idx[32] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
                               8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9};
constexpr int min_in_group[10] = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24};

// sig_coeff_flag context increment (9.3.4.2.5), diagonal scan.
int sig_ctx_inc(int xc, int yc, int log2size, int cidx, int prev_csbf, int xs, int ys)
{
	int sig_ctx;
	if (log2size == 2)
	{
		static constexpr int ctx_idx_map[16] = {0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8};
		sig_ctx = ctx_idx_map[(yc << 2) + xc];
	}
	else if (xc + yc == 0)
	{
		sig_ctx = 0;
	}
	else
	{
		const int xp = xc & 3;
		const int yp = yc & 3;
		switch (prev_csbf)
		{
			case 0: sig_ctx = (xp + yp == 0) ? 2 : (xp + yp < 3) ? 1 : 0; break;
			case 1: sig_ctx = (yp == 0) ? 2 : (yp == 1) ? 1 : 0; break;
			case 2: sig_ctx = (xp == 0) ? 2 : (xp == 1) ? 1 : 0; break;
			default: sig_ctx = 2; break;
		}
		if (cidx == 0)
		{
			if (xs + ys > 0)
				sig_ctx += 3;
			sig_ctx += (log2size == 3) ? 9 : 21; // scanIdx==0 -> +9 for 8x8
		}
		else
		{
			sig_ctx += (log2size == 3) ? 9 : 12;
		}
	}
	return (cidx == 0) ? sig_ctx : 27 + sig_ctx;
}

// coeff_abs_level_remaining binarization (9.3.3.9): concatenated Golomb-Rice /
// Exp-Golomb, all bypass.
void write_coeff_remain(cabac_encoder & cb, int value, int rice)
{
	constexpr int REDUCTION = 3;
	if (value < (REDUCTION << rice))
	{
		const int prefix = value >> rice;
		// prefix ones then a terminating zero (unary), then `rice` suffix bits.
		for (int i = 0; i < prefix; ++i)
			cb.encode_bypass(1);
		cb.encode_bypass(0);
		if (rice > 0)
			cb.encode_bypass_bits(value & ((1 << rice) - 1), rice);
	}
	else
	{
		int rem = value - (REDUCTION << rice);
		int len = rice;
		while (rem >= (1 << len))
		{
			rem -= (1 << len);
			++len;
		}
		// prefix: (REDUCTION + len + 1 - rice) ones then a zero.
		const int prefix_ones = REDUCTION + (len - rice);
		for (int i = 0; i < prefix_ones; ++i)
			cb.encode_bypass(1);
		cb.encode_bypass(0);
		cb.encode_bypass_bits(rem, len);
	}
}

} // namespace

void residual_coding(cabac_encoder & cb, const int32_t * level, int log2size, int cidx)
{
	const int n = 1 << log2size;
	const int nsb = n >> 2; // sub-blocks per side (1 for 4x4, 2 for 8x8)
	const int ctx_cidx = (cidx == 0) ? 0 : 1;

	pos scan16[16];
	diag_scan(4, scan16);
	pos scan_sb[16];
	diag_scan(nsb, scan_sb);
	const int num_sb = nsb * nsb;

	// Locate the last significant coefficient in scan order.
	int last_sb = 0, last_pos_in_sb = 0, last_x = 0, last_y = 0;
	for (int sb = 0; sb < num_sb; ++sb)
	{
		const int xs = scan_sb[sb].x, ys = scan_sb[sb].y;
		for (int p = 0; p < 16; ++p)
		{
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			if (level[cy * n + cx] != 0)
			{
				last_sb = sb;
				last_pos_in_sb = p;
				last_x = cx;
				last_y = cy;
			}
		}
	}

	// last_significant_coeff_x/y prefix (context) + suffix (bypass).
	auto code_last_prefix = [&](int val, int base_ctx) {
		const int gidx = group_idx[val];
		int ctx_offset, ctx_shift;
		if (cidx == 0)
		{
			ctx_offset = 3 * (log2size - 2) + ((log2size - 1) >> 2);
			ctx_shift = (log2size + 1) >> 2;
		}
		else
		{
			ctx_offset = 15;
			ctx_shift = log2size - 2;
		}
		const int cmax = (log2size << 1) - 1;
		for (int b = 0; b < gidx; ++b)
			cb.encode_bin(base_ctx + (b >> ctx_shift) + ctx_offset, 1);
		if (gidx < cmax)
			cb.encode_bin(base_ctx + (gidx >> ctx_shift) + ctx_offset, 0);
	};
	code_last_prefix(last_x, LAST_SIGNIFICANT_COEFF_X_PREFIX);
	code_last_prefix(last_y, LAST_SIGNIFICANT_COEFF_Y_PREFIX);
	auto code_last_suffix = [&](int val) {
		const int gidx = group_idx[val];
		if (gidx >= 4)
		{
			const int nbits = (gidx >> 1) - 1;
			cb.encode_bypass_bits(val - min_in_group[gidx], nbits);
		}
	};
	code_last_suffix(last_x);
	code_last_suffix(last_y);

	// Per-sub-block coefficient coding, from the last sub-block down to DC.
	std::array<std::array<uint8_t, 4>, 4> csbf{}; // coded_sub_block_flag grid
	int c1 = 1; // greater1Ctx carried across sub-blocks

	for (int sb = last_sb; sb >= 0; --sb)
	{
		const int xs = scan_sb[sb].x, ys = scan_sb[sb].y;
		const bool sb_last = (sb == last_sb);
		const bool sb_first = (sb == 0);
		int infer_dc_sig = 0;

		if (not sb_last && not sb_first)
		{
			const int right = (xs + 1 < nsb) ? csbf[ys][xs + 1] : 0;
			const int below = (ys + 1 < nsb) ? csbf[ys + 1][xs] : 0;
			// Does this sub-block contain any significant coefficient?
			int any = 0;
			for (int p = 0; p < 16 && not any; ++p)
				if (level[(ys * 4 + scan16[p].y) * n + (xs * 4 + scan16[p].x)] != 0)
					any = 1;
			cb.encode_bin(SIGNIFICANT_COEFF_GROUP_FLAG + ctx_cidx * 2 + ((right | below) ? 1 : 0), any);
			csbf[ys][xs] = any;
			if (not any)
				continue;
			infer_dc_sig = 1;
		}
		else
		{
			csbf[ys][xs] = 1; // inferred for the first and last sub-blocks
		}

		const int right = (xs + 1 < nsb) ? csbf[ys][xs + 1] : 0;
		const int below = (ys + 1 < nsb) ? csbf[ys + 1][xs] : 0;
		const int prev_csbf = right | (below << 1);

		// sig_coeff_flag for the sub-block's positions, high scan index -> low.
		uint8_t sig[16] = {};
		if (sb_last)
			sig[last_pos_in_sb] = 1; // implied by last_significant_coeff
		const int start = sb_last ? last_pos_in_sb - 1 : 15;
		int coded_any_sig = 0;
		for (int p = start; p >= 0; --p)
		{
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			if (p == 0 && infer_dc_sig && not coded_any_sig)
			{
				sig[0] = 1; // inferred DC significance
				break;
			}
			const int s = (level[cy * n + cx] != 0) ? 1 : 0;
			cb.encode_bin(SIGNIFICANT_COEFF_FLAG + sig_ctx_inc(cx, cy, log2size, cidx, prev_csbf, xs, ys), s);
			sig[p] = s;
			if (s)
				coded_any_sig = 1;
		}

		// Collect significant scan positions, high index -> low.
		int sig_scan[16];
		int num_sig = 0;
		for (int p = 15; p >= 0; --p)
			if (sig[p])
				sig_scan[num_sig++] = p;
		if (num_sig == 0)
			continue;

		int ctx_set = (sb > 0 && cidx == 0) ? 2 : 0;
		if (c1 == 0)
			++ctx_set;
		c1 = 1;

		// coeff_abs_level_greater1_flag for up to the first 8 significant coeffs.
		const int num_gt1 = num_sig < 8 ? num_sig : 8;
		uint8_t gt1[16] = {};
		int first_gt1 = -1;
		for (int k = 0; k < num_gt1; ++k)
		{
			const int p = sig_scan[k];
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			const int b = (std::abs(level[cy * n + cx]) > 1) ? 1 : 0;
			cb.encode_bin(COEFF_ABS_LEVEL_GREATER1_FLAG + ctx_cidx * 16 + ctx_set * 4 + c1, b);
			gt1[k] = b;
			if (b)
			{
				if (first_gt1 < 0)
					first_gt1 = k;
				c1 = 0;
			}
			else if (c1 < 3 && c1 > 0)
			{
				++c1;
			}
		}

		// coeff_abs_level_greater2_flag for the first greater1 coeff only.
		int gt2 = 0;
		if (first_gt1 >= 0)
		{
			const int p = sig_scan[first_gt1];
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			gt2 = (std::abs(level[cy * n + cx]) > 2) ? 1 : 0;
			cb.encode_bin(COEFF_ABS_LEVEL_GREATER2_FLAG + ctx_cidx * 4 + ctx_set, gt2);
		}

		// coeff_sign_flag (bypass) for every significant coeff, high -> low.
		for (int k = 0; k < num_sig; ++k)
		{
			const int p = sig_scan[k];
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			cb.encode_bypass(level[cy * n + cx] < 0 ? 1 : 0);
		}

		// coeff_abs_level_remaining (bypass), high -> low, adaptive rice.
		int rice = 0;
		for (int k = 0; k < num_sig; ++k)
		{
			const int p = sig_scan[k];
			const int cx = xs * 4 + scan16[p].x;
			const int cy = ys * 4 + scan16[p].y;
			const int absv = std::abs(level[cy * n + cx]);

			int base_level;
			if (k >= 8)
				base_level = 1;
			else if (gt1[k] == 0)
				base_level = -1; // no remaining (level fully known == 1)
			else if (k == first_gt1)
				base_level = gt2 ? 3 : -1; // gt2==0 -> level == 2, no remaining
			else
				base_level = 2;

			if (base_level >= 0 && absv >= base_level)
			{
				write_coeff_remain(cb, absv - base_level, rice);
				if (absv > (3 << rice))
					rice = rice < 4 ? rice + 1 : 4;
			}
		}
	}
}

} // namespace wivrn::hevc
