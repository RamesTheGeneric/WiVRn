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

#include "hevc_tables.h"

#include <cstdint>
#include <vector>

namespace wivrn::hevc
{

// Reference CABAC arithmetic encoder (Rec. ITU-T H.265 clause 9.3). This is the
// CPU "golden" coder: it defines exactly the byte stream a conforming decoder
// expects, and the GPU CABAC compute kernel will be validated byte-for-byte
// against it. The arithmetic engine is identical to H.264 Annex; only the
// context models and their initialisation are HEVC-specific.
class cabac_encoder
{
	// Per-context adaptive probability state.
	uint8_t pstate[tables::HEVC_CONTEXTS]; // pStateIdx in [0, 62]
	uint8_t valmps[tables::HEVC_CONTEXTS]; // most-probable-symbol value

	// Arithmetic engine state (clause 9.3.4.3).
	int range = 510;
	int low = 0;
	int bits_outstanding = 0;
	bool first_bit = true;

	// Bit sink (MSB first) accumulating the CABAC-coded slice data bytes.
	std::vector<uint8_t> out;
	uint32_t cur = 0;
	int nbits = 0;

	void put_bit_raw(int b);
	void put_bit(int b); // handles outstanding bits (clause 9.3.4.3.5)
	void renorm();       // RenormE (clause 9.3.4.3.3)

public:
	// Initialise all contexts for the given initType (0 = I slice) and slice QP,
	// and reset the arithmetic engine (clause 9.3.2.2 / 9.3.4.3.1).
	void init(int init_type, int slice_qp_y);

	// encodeDecision: code one context-modelled bin (clause 9.3.4.3.2).
	void encode_bin(int ctx_idx, int bin);

	// encodeBypass: code one equiprobable bin (clause 9.3.4.3.4).
	void encode_bypass(int bin);

	// Code the low `n` bits of `value`, MSB first, in bypass mode.
	void encode_bypass_bits(uint32_t value, int n);

	// encodeTerminate: code end_of_sub_stream/slice or pcm alignment
	// (clause 9.3.4.3.5). bin == 1 finalises the stream.
	void encode_terminate(int bin);

	// Flush the engine after a terminate(1) and byte-align, returning the coded
	// slice-data bytes (byte aligned, ready to append after the slice header).
	std::vector<uint8_t> finish();

	size_t bit_count() const { return out.size() * 8 + nbits; }
};

} // namespace wivrn::hevc
