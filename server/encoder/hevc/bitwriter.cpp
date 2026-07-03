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

#include "bitwriter.h"

#include <cassert>

namespace wivrn::hevc
{

void bitwriter::put_bit(uint32_t bit)
{
	assert(bit <= 1);
	cur = (cur << 1) | (bit & 1u);
	++nbits;
	if (nbits == 8)
	{
		buf.push_back(static_cast<uint8_t>(cur & 0xff));
		cur = 0;
		nbits = 0;
	}
}

void bitwriter::put_bits(uint32_t value, int n)
{
	assert(n >= 0 && n <= 32);
	for (int i = n - 1; i >= 0; --i)
		put_bit((value >> i) & 1u);
}

void bitwriter::put_ue(uint32_t value)
{
	// codeNum = value; write prefix of `leading_zeros` zeros, a 1, then the
	// `leading_zeros` low bits of (value + 1).
	uint32_t code = value + 1; // in [1, 2^32]; value==UINT32_MAX would overflow
	int leading_zeros = 0;
	// Number of bits needed to represent `code`.
	uint32_t tmp = code;
	int nbits_needed = 0;
	while (tmp)
	{
		tmp >>= 1;
		++nbits_needed;
	}
	leading_zeros = nbits_needed - 1;
	for (int i = 0; i < leading_zeros; ++i)
		put_bit(0);
	// Emit `code` in `nbits_needed` bits (its MSB is the separating 1 bit).
	put_bits(code, nbits_needed);
}

void bitwriter::put_se(int32_t value)
{
	uint32_t mapped;
	if (value > 0)
		mapped = static_cast<uint32_t>(value) * 2u - 1u;
	else
		mapped = static_cast<uint32_t>(-value) * 2u;
	put_ue(mapped);
}

void bitwriter::rbsp_trailing_bits()
{
	put_bit(1); // rbsp_stop_one_bit
	while (nbits != 0)
		put_bit(0); // rbsp_alignment_zero_bit
}

void bitwriter::byte_alignment()
{
	put_bit(1); // alignment_bit_equal_to_one
	while (nbits != 0)
		put_bit(0); // alignment_bit_equal_to_zero
}

const std::vector<uint8_t> & bitwriter::bytes()
{
	if (nbits != 0)
	{
		// Zero-pad the pending partial byte to a full byte.
		cur <<= (8 - nbits);
		buf.push_back(static_cast<uint8_t>(cur & 0xff));
		cur = 0;
		nbits = 0;
	}
	return buf;
}

void append_ebsp(std::vector<uint8_t> & out, const std::vector<uint8_t> & rbsp)
{
	int zeros = 0;
	for (uint8_t b: rbsp)
	{
		if (zeros >= 2 && b <= 0x03)
		{
			out.push_back(0x03);
			zeros = 0;
		}
		out.push_back(b);
		if (b == 0x00)
			++zeros;
		else
			zeros = 0;
	}
}

void emit_nal(std::vector<uint8_t> & out, int nal_unit_type, const std::vector<uint8_t> & rbsp)
{
	// Annex-B start code.
	out.push_back(0x00);
	out.push_back(0x00);
	out.push_back(0x00);
	out.push_back(0x01);
	// HEVC nal_unit_header (2 bytes): forbidden_zero_bit(1)=0,
	// nal_unit_type(6), nuh_layer_id(6)=0, nuh_temporal_id_plus1(3)=1.
	out.push_back(static_cast<uint8_t>((nal_unit_type & 0x3f) << 1));
	out.push_back(0x01);
	append_ebsp(out, rbsp);
}

} // namespace wivrn::hevc
