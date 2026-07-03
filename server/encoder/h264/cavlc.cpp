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

#include "cavlc.h"

#include <cstdlib>

namespace wivrn::h264
{

const int zigzag4x4[16] = {0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15};

namespace
{
// CAVLC VLC tables (Rec. ITU-T H.264 Tables 9-5, 9-7, 9-8, 9-10), transcribed
// verbatim from the reference layout: index = TotalCoeff*4 + TrailingOnes.
const uint8_t coeff_token_len[4][4 * 17] = {
        {
                1, 0, 0, 0,
                6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6,
                11, 10, 9, 7, 13, 11, 10, 8, 13, 13, 11, 9, 13, 13, 13, 10,
                14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14,
                16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16},
        {
                2, 0, 0, 0,
                6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4,
                8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9, 6, 11, 11, 11, 7,
                12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12,
                13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14},
        {
                4, 0, 0, 0,
                6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4,
                7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4, 8, 7, 7, 5,
                8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8,
                10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
        {
                6, 0, 0, 0,
                6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}};

const uint8_t coeff_token_bits[4][4 * 17] = {
        {
                1, 0, 0, 0,
                5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3,
                7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5, 4, 8, 10, 13, 4,
                15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8,
                15, 1, 9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8},
        {
                3, 0, 0, 0,
                11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4,
                4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5, 4, 11, 14, 13, 4,
                15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12,
                11, 10, 9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4},
        {
                15, 0, 0, 0,
                15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11,
                11, 8, 9, 10, 9, 14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13,
                11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13, 8,
                13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2},
        {
                3, 0, 0, 0,
                0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63}};

const uint8_t chroma_dc_coeff_token_len[4 * 5] = {
        2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7};
const uint8_t chroma_dc_coeff_token_bits[4 * 5] = {
        1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0};

const uint8_t total_zeros_len[15][16] = {
        {1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9},
        {3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6},
        {4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6},
        {5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5},
        {4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5},
        {6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6},
        {6, 5, 3, 3, 3, 2, 3, 4, 3, 6},
        {6, 4, 5, 3, 2, 2, 3, 3, 6},
        {6, 6, 4, 2, 2, 3, 2, 5},
        {5, 5, 3, 2, 2, 2, 4},
        {4, 4, 3, 3, 1, 3},
        {4, 4, 2, 1, 3},
        {3, 3, 1, 2},
        {2, 2, 1},
        {1, 1}};
const uint8_t total_zeros_bits[15][16] = {
        {1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1},
        {7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0},
        {5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0},
        {3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0},
        {5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0},
        {1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0},
        {1, 1, 5, 4, 3, 3, 2, 1, 1, 0},
        {1, 1, 1, 3, 3, 2, 2, 1, 0},
        {1, 0, 1, 3, 2, 1, 1, 1},
        {1, 0, 1, 3, 2, 1, 1},
        {0, 1, 1, 2, 1, 3},
        {0, 1, 1, 1, 1},
        {0, 1, 1, 1},
        {0, 1, 1},
        {0, 1}};

const uint8_t chroma_dc_total_zeros_len[3][4] = {{1, 2, 3, 3}, {1, 2, 2, 0}, {1, 1, 0, 0}};
const uint8_t chroma_dc_total_zeros_bits[3][4] = {{1, 1, 1, 0}, {1, 1, 0, 0}, {1, 0, 0, 0}};

const uint8_t run_len[7][16] = {
        {1, 1},
        {1, 2, 2},
        {2, 2, 2, 2},
        {2, 2, 2, 3, 3},
        {2, 2, 3, 3, 3, 3},
        {2, 3, 3, 3, 3, 3, 3},
        {3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
const uint8_t run_bits[7][16] = {
        {1, 0},
        {1, 1, 0},
        {3, 2, 1, 0},
        {3, 2, 1, 1, 0},
        {3, 2, 3, 2, 1, 0},
        {3, 0, 1, 3, 2, 5, 4},
        {7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

void write_coeff_token(bitwriter & w, int total_coeff, int trailing_ones, int nC, bool chroma_dc)
{
	const int idx = total_coeff * 4 + trailing_ones;
	if (chroma_dc)
	{
		w.put_bits(chroma_dc_coeff_token_bits[idx], chroma_dc_coeff_token_len[idx]);
		return;
	}
	int t;
	if (nC < 2)
		t = 0;
	else if (nC < 4)
		t = 1;
	else if (nC < 8)
		t = 2;
	else
		t = 3;
	w.put_bits(coeff_token_bits[t][idx], coeff_token_len[t][idx]);
}

// Emit one level's level_prefix (unary) + level_suffix for the given level_code
// and suffix length (inverse of the decoder in clause 9.2.2.1).
void write_level(bitwriter & w, int level_code, int suffix_length)
{
	int prefix, suffix = 0, suffix_size = 0;
	if (suffix_length == 0)
	{
		if (level_code < 14)
		{
			prefix = level_code;
		}
		else if (level_code < 30)
		{
			prefix = 14;
			suffix = level_code - 14;
			suffix_size = 4;
		}
		else
		{
			prefix = 15;
			suffix = level_code - 30;
			suffix_size = 12;
		}
	}
	else
	{
		prefix = level_code >> suffix_length;
		if (prefix < 15)
		{
			suffix = level_code & ((1 << suffix_length) - 1);
			suffix_size = suffix_length;
		}
		else
		{
			prefix = 15;
			suffix = level_code - (15 << suffix_length);
			suffix_size = 12;
		}
	}
	for (int i = 0; i < prefix; ++i)
		w.put_bit(0);
	w.put_bit(1);
	if (suffix_size > 0)
		w.put_bits((uint32_t)suffix, suffix_size);
}
} // namespace

int residual_block(bitwriter & w, const int * levels, int count, int nC, bool chroma_dc)
{
	// Collect non-zero coefficients from high frequency to low.
	int run[16] = {};    // zeros immediately before each nonzero (high->low)
	int lvl[16] = {};    // signed level (high->low)
	int total_coeff = 0;
	int total_zeros = 0; // zeros before the last (highest-freq) nonzero
	int last_nz = -1;
	for (int i = count - 1; i >= 0; --i)
		if (levels[i] != 0)
		{
			last_nz = i;
			break;
		}
	if (last_nz >= 0)
	{
		int run_acc = 0;
		for (int i = last_nz; i >= 0; --i)
		{
			if (levels[i] != 0)
			{
				lvl[total_coeff] = levels[i];
				// run_before[k] is the zeros between coeff k and the next lower
				// coeff k+1; when we reach coeff `total_coeff`, the zeros since the
				// previous (higher) coeff are that previous coeff's run_before.
				if (total_coeff >= 1)
					run[total_coeff - 1] = run_acc;
				++total_coeff;
				run_acc = 0;
			}
			else
			{
				++run_acc;
			}
		}
		total_zeros = last_nz + 1 - total_coeff;
	}

	// TrailingOnes: leading (highest-freq) coefficients with magnitude 1, up to 3.
	int trailing_ones = 0;
	while (trailing_ones < total_coeff && trailing_ones < 3 && std::abs(lvl[trailing_ones]) == 1)
		++trailing_ones;

	write_coeff_token(w, total_coeff, trailing_ones, nC, chroma_dc);
	if (total_coeff == 0)
		return 0;

	// trailing_ones_sign_flag, high freq -> low.
	for (int i = 0; i < trailing_ones; ++i)
		w.put_bit(lvl[i] < 0 ? 1 : 0);

	// Remaining levels, high freq -> low.
	int suffix_length = (total_coeff > 10 && trailing_ones < 3) ? 1 : 0;
	for (int i = trailing_ones; i < total_coeff; ++i)
	{
		const int level = lvl[i];
		int level_code = (level > 0) ? ((level << 1) - 2) : ((-level << 1) - 1);
		if (i == trailing_ones && trailing_ones < 3)
			level_code -= 2;
		write_level(w, level_code, suffix_length);
		if (suffix_length == 0)
			suffix_length = 1;
		if (std::abs(level) > (3 << (suffix_length - 1)) && suffix_length < 6)
			++suffix_length;
	}

	// total_zeros (only if the block isn't full).
	if (total_coeff < count)
	{
		if (chroma_dc)
			w.put_bits(chroma_dc_total_zeros_bits[total_coeff - 1][total_zeros],
			           chroma_dc_total_zeros_len[total_coeff - 1][total_zeros]);
		else
			w.put_bits(total_zeros_bits[total_coeff - 1][total_zeros],
			           total_zeros_len[total_coeff - 1][total_zeros]);
	}

	// run_before for every nonzero except the last (lowest freq), high -> low.
	int zeros_left = total_zeros;
	for (int i = 0; i < total_coeff - 1; ++i)
	{
		if (zeros_left <= 0)
			break;
		const int t = (zeros_left < 7 ? zeros_left : 7) - 1;
		const int rb = run[i];
		w.put_bits(run_bits[t][rb], run_len[t][rb]);
		zeros_left -= rb;
	}

	return total_coeff;
}

} // namespace wivrn::h264
