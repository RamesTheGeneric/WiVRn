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
#include <vector>

namespace wivrn::h267
{

// Bit-level writer producing a Raw Byte Sequence Payload (RBSP), i.e. the
// bytes of a NAL unit *before* emulation-prevention and start codes are added.
//
// Bits are packed MSB-first within each byte, as required by the H.265/HEVC
// (and H.264) bitstream syntax. All syntax primitives from Rec. ITU-T H.265
// clause 9.2 (Exp-Golomb) and clause 7 (fixed-length) are provided.
class bitwriter
{
	std::vector<uint8_t> buf;
	uint32_t cur = 0;   // partially filled byte, left-aligned in the low `nbits` bits
	int nbits = 0;      // number of valid bits currently held in `cur` (0..7)

public:
	bitwriter() = default;

	// u(1): append a single bit (value must be 0 or 1).
	void put_bit(uint32_t bit);

	// u(n): append the low `n` bits of `value`, MSB first. 0 <= n <= 32.
	void put_bits(uint32_t value, int n);

	// ue(v): unsigned Exp-Golomb (clause 9.2).
	void put_ue(uint32_t value);

	// se(v): signed Exp-Golomb (clause 9.2).
	void put_se(int32_t value);

	// True when no partial byte is pending.
	bool byte_aligned() const { return nbits == 0; }

	// rbsp_trailing_bits(): a stop bit (1) followed by zero bits until byte
	// aligned (clause 7.3.2.11). Every parameter-set / slice RBSP ends with this.
	void rbsp_trailing_bits();

	// byte_alignment(): a 1 bit then 0 bits to the next byte boundary
	// (clause 7.3.2.9). Used inside slice segment headers before byte-aligned
	// substream data.
	void byte_alignment();

	// cabac_alignment_one_bits() emitted before a CABAC-aligned region is the
	// same shape; provided for readability at call sites.
	void alignment_one_bits() { byte_alignment(); }

	// Number of bits written so far (including any pending partial byte).
	size_t bit_count() const { return buf.size() * 8 + nbits; }

	// Finalize any pending partial byte by zero-padding to a byte boundary and
	// return the accumulated RBSP bytes. Call after rbsp_trailing_bits() (which
	// already aligns) or when raw byte output is wanted.
	const std::vector<uint8_t> & bytes();

	// Access without finalizing (must be byte aligned).
	const std::vector<uint8_t> & raw() const { return buf; }
};

// Apply emulation-prevention (clause 7.4.2): insert an emulation_prevention_
// three_byte (0x03) whenever a sequence 0x00 0x00 0x00/0x01/0x02/0x03 would
// otherwise appear in the RBSP, turning it into an EBSP. Appends to `out`.
void append_ebsp(std::vector<uint8_t> & out, const std::vector<uint8_t> & rbsp);

// Emit a complete Annex-B NAL unit into `out`: a 4-byte start code
// (00 00 00 01), the 2-byte HEVC NAL unit header for `nal_unit_type` (with
// nuh_layer_id = 0 and nuh_temporal_id_plus1 = 1), then the emulation-prevented
// RBSP. `rbsp` must already be byte aligned (i.e. end with rbsp_trailing_bits).
void emit_nal(std::vector<uint8_t> & out, int nal_unit_type, const std::vector<uint8_t> & rbsp);

} // namespace wivrn::h267
