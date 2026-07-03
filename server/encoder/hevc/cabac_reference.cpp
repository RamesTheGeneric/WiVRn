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

#include "cabac_reference.h"

#include <algorithm>

namespace wivrn::hevc
{

using namespace tables;

namespace
{
int clip3(int lo, int hi, int v)
{
	return v < lo ? lo : (v > hi ? hi : v);
}
} // namespace

void cabac_encoder::init(int init_type, int slice_qp_y)
{
	const int qp = clip3(0, 51, slice_qp_y);
	for (int i = 0; i < HEVC_CONTEXTS; ++i)
	{
		const int init_value = init_values[init_type][i];
		const int slope_idx = init_value >> 4;
		const int offset_idx = init_value & 15;
		const int m = slope_idx * 5 - 45;
		const int n = (offset_idx << 3) - 16;
		const int pre_ctx_state = clip3(1, 126, ((m * qp) >> 4) + n);
		if (pre_ctx_state <= 63)
		{
			pstate[i] = static_cast<uint8_t>(63 - pre_ctx_state);
			valmps[i] = 0;
		}
		else
		{
			pstate[i] = static_cast<uint8_t>(pre_ctx_state - 64);
			valmps[i] = 1;
		}
	}

	range = 510;
	low = 0;
	bits_outstanding = 0;
	first_bit = true;
	out.clear();
	cur = 0;
	nbits = 0;
}

void cabac_encoder::put_bit_raw(int b)
{
	cur = (cur << 1) | (b & 1);
	if (++nbits == 8)
	{
		out.push_back(static_cast<uint8_t>(cur & 0xff));
		cur = 0;
		nbits = 0;
	}
}

void cabac_encoder::put_bit(int b)
{
	// The very first bit produced is a leading bit that is not emitted
	// (firstBitFlag), matching the decoder's initialisation read.
	if (first_bit)
		first_bit = false;
	else
		put_bit_raw(b);
	while (bits_outstanding > 0)
	{
		put_bit_raw(1 - b);
		--bits_outstanding;
	}
}

void cabac_encoder::renorm()
{
	while (range < 256)
	{
		if (low < 256)
		{
			put_bit(0);
		}
		else if (low >= 512)
		{
			low -= 512;
			put_bit(1);
		}
		else
		{
			low -= 256;
			++bits_outstanding;
		}
		range <<= 1;
		low <<= 1;
	}
}

void cabac_encoder::encode_bin(int ctx_idx, int bin)
{
	const int q = (range >> 6) & 3;
	const int lps = rangeTabLps[pstate[ctx_idx]][q];
	range -= lps;
	if (bin != valmps[ctx_idx])
	{
		low += range;
		range = lps;
		if (pstate[ctx_idx] == 0)
			valmps[ctx_idx] = 1 - valmps[ctx_idx];
		pstate[ctx_idx] = transIdxLps[pstate[ctx_idx]];
	}
	else
	{
		pstate[ctx_idx] = transIdxMps[pstate[ctx_idx]];
	}
	renorm();
}

void cabac_encoder::encode_bypass(int bin)
{
	low <<= 1;
	if (bin)
		low += range;
	if (low >= 1024)
	{
		put_bit(1);
		low -= 1024;
	}
	else if (low < 512)
	{
		put_bit(0);
	}
	else
	{
		low -= 512;
		++bits_outstanding;
	}
}

void cabac_encoder::encode_bypass_bits(uint32_t value, int n)
{
	for (int i = n - 1; i >= 0; --i)
		encode_bypass((value >> i) & 1);
}

void cabac_encoder::encode_terminate(int bin)
{
	range -= 2;
	if (bin)
	{
		low += range;
		// Flush (clause 9.3.4.3.5 EncodeFlush).
		range = 2;
		renorm();
		put_bit((low >> 9) & 1);
		// Final two bits, with rbsp stop bit folded in via | 1.
		put_bit_raw((low >> 8) & 1);
		put_bit_raw(1);
	}
	else
	{
		renorm();
	}
}

std::vector<uint8_t> cabac_encoder::finish()
{
	// Byte-align the coded slice data (rbsp_slice_segment_trailing_bits: the
	// stop bit was folded into the terminate flush, so pad with zeros).
	while (nbits != 0)
		put_bit_raw(0);
	return std::move(out);
}

} // namespace wivrn::hevc
