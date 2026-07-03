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

#include "transform.h"

#include <cstdlib>

namespace wivrn::hevc::xform
{

namespace
{

// transMatrix[k][n]: DCT-II basis, k = frequency, n = spatial position.
// These are the canonical HEVC integer matrices (Rec. ITU-T H.265 8.6.4.2).
constexpr int dct4[4][4] = {
        {64, 64, 64, 64},
        {83, 36, -36, -83},
        {64, -64, -64, 64},
        {36, -83, 83, -36},
};
constexpr int dct8[8][8] = {
        {64, 64, 64, 64, 64, 64, 64, 64},
        {89, 75, 50, 18, -18, -50, -75, -89},
        {83, 36, -36, -83, -83, -36, 36, 83},
        {75, -18, -89, -50, 50, 89, 18, -75},
        {64, -64, -64, 64, 64, -64, -64, 64},
        {50, -89, 18, 75, -75, -18, 89, -50},
        {36, -83, 83, -36, -36, 83, -83, 36},
        {18, -50, 75, -89, 89, -75, 50, -18},
};

int mat(int n, int k, int i)
{
	return n == 4 ? dct4[k][i] : dct8[k][i];
}

int log2i(int n)
{
	return n == 4 ? 2 : 3; // only 4 and 8 supported here
}

int clip16(int v)
{
	if (v < -32768)
		return -32768;
	if (v > 32767)
		return 32767;
	return v;
}

constexpr int level_scale[6] = {40, 45, 51, 57, 64, 72};
constexpr int quant_scale[6] = {26214, 23302, 20560, 18396, 16384, 14564};

} // namespace

void idct(const int32_t * in, int32_t * out, int n, int bit_depth)
{
	const int log2n = log2i(n);
	int tmp[8 * 8];

	// First (vertical) inverse pass: e[y][x] = sum_k M[k][y] * d[k][x], >>7.
	for (int x = 0; x < n; ++x)
	{
		for (int y = 0; y < n; ++y)
		{
			int acc = 0;
			for (int k = 0; k < n; ++k)
				acc += mat(n, k, y) * in[k * n + x];
			tmp[y * n + x] = clip16((acc + 64) >> 7);
		}
	}

	// Second (horizontal) inverse pass: r[y][x] = sum_k M[k][x] * g[y][k],
	// >> (20 - bitDepth).
	const int shift2 = 20 - bit_depth;
	const int add2 = 1 << (shift2 - 1);
	for (int y = 0; y < n; ++y)
	{
		for (int x = 0; x < n; ++x)
		{
			int acc = 0;
			for (int k = 0; k < n; ++k)
				acc += mat(n, k, x) * tmp[y * n + k];
			out[y * n + x] = (acc + add2) >> shift2;
		}
	}
	(void)log2n;
}

void fdct(const int32_t * in, int32_t * out, int n, int bit_depth)
{
	const int log2n = log2i(n);
	int tmp[8 * 8];

	// First (horizontal) forward pass: a[y][k] = sum_n M[k][n] * src[y][n].
	const int shift1 = log2n + bit_depth - 9;
	const int add1 = shift1 > 0 ? (1 << (shift1 - 1)) : 0;
	for (int y = 0; y < n; ++y)
	{
		for (int k = 0; k < n; ++k)
		{
			int acc = 0;
			for (int i = 0; i < n; ++i)
				acc += mat(n, k, i) * in[y * n + i];
			tmp[y * n + k] = shift1 > 0 ? ((acc + add1) >> shift1) : acc;
		}
	}

	// Second (vertical) forward pass: b[k2][k] = sum_y M[k2][y] * a[y][k].
	const int shift2 = log2n + 6;
	const int add2 = 1 << (shift2 - 1);
	for (int k = 0; k < n; ++k)
	{
		for (int k2 = 0; k2 < n; ++k2)
		{
			int acc = 0;
			for (int y = 0; y < n; ++y)
				acc += mat(n, k2, y) * tmp[y * n + k];
			out[k2 * n + k] = (acc + add2) >> shift2;
		}
	}
}

void dequant(const int32_t * level, int32_t * coeff, int n, int qp, int bit_depth)
{
	const int log2n = log2i(n);
	const int bd_shift = bit_depth + log2n - 5;
	const int add = 1 << (bd_shift - 1);
	const int scale = level_scale[qp % 6];
	const int shift = qp / 6;
	const int m = 16; // flat scaling list
	for (int i = 0; i < n * n; ++i)
	{
		long v = ((long)level[i] * m * scale << shift);
		v = (v + add) >> bd_shift;
		coeff[i] = clip16((int)v);
	}
}

void quant(const int32_t * coeff, int32_t * level, int n, int qp, int bit_depth)
{
	const int log2n = log2i(n);
	const int transform_shift = 15 - bit_depth - log2n;
	const int qbits = 14 + qp / 6 + transform_shift;
	const int scale = quant_scale[qp % 6];
	// Intra dead-zone rounding offset (HM: 171/512 of the quant step).
	const long offset = ((long)171 << qbits) / 512;
	for (int i = 0; i < n * n; ++i)
	{
		const int c = coeff[i];
		const long a = (long)std::abs(c) * scale;
		int lvl = (int)((a + offset) >> qbits);
		level[i] = c < 0 ? -lvl : lvl;
	}
}

} // namespace wivrn::hevc::xform
