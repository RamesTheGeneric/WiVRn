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

#include "transform264.h"

#include <cstdlib>

namespace wivrn::avc::xform
{

namespace
{
// Forward quant multipliers MF[qp%6][class] and dequant LevelScale V[qp%6][class].
constexpr int MF[6][3] = {
        {13107, 5243, 8066}, {11916, 4660, 7490}, {10082, 4194, 6554}, {9362, 3647, 5825}, {8192, 3355, 5243}, {7282, 2893, 4559}};
constexpr int V[6][3] = {
        {10, 16, 13}, {11, 18, 14}, {13, 20, 16}, {14, 23, 18}, {16, 25, 20}, {18, 29, 23}};

// Position class for a 4x4 coefficient at (row, col).
int cls(int i, int j)
{
	if ((i % 2 == 0) && (j % 2 == 0))
		return 0;
	if ((i % 2 == 1) && (j % 2 == 1))
		return 1;
	return 2;
}
} // namespace

int chroma_qp(int qp)
{
	static const int t[52] = {
	        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
	        26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39};
	if (qp < 0)
		qp = 0;
	if (qp > 51)
		qp = 51;
	return t[qp];
}

void fdct4(const int * in, int * out)
{
	int tmp[16];
	// rows
	for (int i = 0; i < 4; ++i)
	{
		const int * x = in + i * 4;
		int t0 = x[0] + x[3];
		int t1 = x[1] + x[2];
		int t2 = x[1] - x[2];
		int t3 = x[0] - x[3];
		tmp[i * 4 + 0] = t0 + t1;
		tmp[i * 4 + 1] = 2 * t3 + t2;
		tmp[i * 4 + 2] = t0 - t1;
		tmp[i * 4 + 3] = t3 - 2 * t2;
	}
	// columns
	for (int j = 0; j < 4; ++j)
	{
		int t0 = tmp[0 * 4 + j] + tmp[3 * 4 + j];
		int t1 = tmp[1 * 4 + j] + tmp[2 * 4 + j];
		int t2 = tmp[1 * 4 + j] - tmp[2 * 4 + j];
		int t3 = tmp[0 * 4 + j] - tmp[3 * 4 + j];
		out[0 * 4 + j] = t0 + t1;
		out[1 * 4 + j] = 2 * t3 + t2;
		out[2 * 4 + j] = t0 - t1;
		out[3 * 4 + j] = t3 - 2 * t2;
	}
}

void fhadamard4(const int * in, int * out)
{
	// Forward luma DC Hadamard (x264 dct4x4dc): the second pass carries a
	// (x + 1) >> 1 normalisation so the produced levels are the correct magnitude
	// for the decoder's (unshifted) inverse Hadamard in clause 8.5.10.
	int tmp[16];
	for (int i = 0; i < 4; ++i)
	{
		const int * x = in + i * 4;
		int t0 = x[0] + x[3];
		int t1 = x[1] + x[2];
		int t2 = x[1] - x[2];
		int t3 = x[0] - x[3];
		tmp[i * 4 + 0] = t0 + t1;
		tmp[i * 4 + 1] = t3 + t2;
		tmp[i * 4 + 2] = t0 - t1;
		tmp[i * 4 + 3] = t3 - t2;
	}
	for (int j = 0; j < 4; ++j)
	{
		int t0 = tmp[0 * 4 + j] + tmp[3 * 4 + j];
		int t1 = tmp[1 * 4 + j] + tmp[2 * 4 + j];
		int t2 = tmp[1 * 4 + j] - tmp[2 * 4 + j];
		int t3 = tmp[0 * 4 + j] - tmp[3 * 4 + j];
		out[0 * 4 + j] = (t0 + t1 + 1) >> 1;
		out[1 * 4 + j] = (t3 + t2 + 1) >> 1;
		out[2 * 4 + j] = (t0 - t1 + 1) >> 1;
		out[3 * 4 + j] = (t3 - t2 + 1) >> 1;
	}
}

void fhadamard2(const int * c, int * o)
{
	// 2x2 Hadamard: Hc * C * Hc, Hc = [[1,1],[1,-1]].
	o[0] = c[0] + c[1] + c[2] + c[3];
	o[1] = c[0] - c[1] + c[2] - c[3];
	o[2] = c[0] + c[1] - c[2] - c[3];
	o[3] = c[0] - c[1] - c[2] + c[3];
}

int quant_ac(int coeff, int row, int col, int qp, bool intra)
{
	const int qbits = 15 + qp / 6;
	const int mf = MF[qp % 6][cls(row, col)];
	const int f = intra ? (1 << qbits) / 3 : (1 << qbits) / 6;
	const int a = std::abs(coeff);
	const int lvl = (a * mf + f) >> qbits;
	return coeff < 0 ? -lvl : lvl;
}

int quant_dc_luma(int had, int qp, bool intra)
{
	const int qbits = 15 + qp / 6;
	const int mf = MF[qp % 6][0];
	const int f = intra ? (1 << qbits) / 3 : (1 << qbits) / 6;
	const int a = std::abs(had);
	const int lvl = (a * mf + 2 * f) >> (qbits + 1);
	return had < 0 ? -lvl : lvl;
}

int quant_dc_chroma(int had, int qpc, bool intra)
{
	const int qbits = 15 + qpc / 6;
	const int mf = MF[qpc % 6][0];
	const int f = intra ? (1 << qbits) / 3 : (1 << qbits) / 6;
	const int a = std::abs(had);
	const int lvl = (a * mf + 2 * f) >> (qbits + 1);
	return had < 0 ? -lvl : lvl;
}

namespace
{
void inv_hadamard4(const int * c, int * f)
{
	int tmp[16];
	for (int i = 0; i < 4; ++i)
	{
		const int * x = c + i * 4;
		tmp[i * 4 + 0] = x[0] + x[1] + x[2] + x[3];
		tmp[i * 4 + 1] = x[0] + x[1] - x[2] - x[3];
		tmp[i * 4 + 2] = x[0] - x[1] - x[2] + x[3];
		tmp[i * 4 + 3] = x[0] - x[1] + x[2] - x[3];
	}
	for (int j = 0; j < 4; ++j)
	{
		int a = tmp[0 * 4 + j], b = tmp[1 * 4 + j], c2 = tmp[2 * 4 + j], d = tmp[3 * 4 + j];
		f[0 * 4 + j] = a + b + c2 + d;
		f[1 * 4 + j] = a + b - c2 - d;
		f[2 * 4 + j] = a - b - c2 + d;
		f[3 * 4 + j] = a - b + c2 - d;
	}
}
} // namespace

void idc_luma(const int * level, int qp, int * dcY)
{
	int f[16];
	inv_hadamard4(level, f);
	const int ls = 16 * V[qp % 6][0];
	for (int i = 0; i < 16; ++i)
	{
		if (qp >= 36)
			dcY[i] = (f[i] * ls) << (qp / 6 - 6);
		else
			dcY[i] = (f[i] * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6);
	}
}

void idc_chroma(const int * level, int qpc, int * dcC)
{
	// inverse 2x2 Hadamard (Hc * Z * Hc)
	int f[4];
	f[0] = level[0] + level[1] + level[2] + level[3];
	f[1] = level[0] - level[1] + level[2] - level[3];
	f[2] = level[0] + level[1] - level[2] - level[3];
	f[3] = level[0] - level[1] - level[2] + level[3];
	const int ls = 16 * V[qpc % 6][0];
	for (int i = 0; i < 4; ++i)
		dcC[i] = ((f[i] * ls) << (qpc / 6)) >> 5;
}

void idct4(const int * level, int qp, bool use_dc, int dc_override, int * resid)
{
	// 8.5.12.1 scaling
	int d[16];
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
		{
			const int idx = i * 4 + j;
			if (use_dc && idx == 0)
			{
				d[0] = dc_override;
				continue;
			}
			const int ls = 16 * V[qp % 6][cls(i, j)];
			if (qp >= 24)
				d[idx] = (level[idx] * ls) << (qp / 6 - 4);
			else
				d[idx] = (level[idx] * ls + (1 << (3 - qp / 6))) >> (4 - qp / 6);
		}

	// 8.5.12.2 inverse core transform
	int f[16];
	for (int i = 0; i < 4; ++i)
	{
		const int * r = d + i * 4;
		int e0 = r[0] + r[2];
		int e1 = r[0] - r[2];
		int e2 = (r[1] >> 1) - r[3];
		int e3 = r[1] + (r[3] >> 1);
		f[i * 4 + 0] = e0 + e3;
		f[i * 4 + 1] = e1 + e2;
		f[i * 4 + 2] = e1 - e2;
		f[i * 4 + 3] = e0 - e3;
	}
	for (int j = 0; j < 4; ++j)
	{
		int g0 = f[0 * 4 + j] + f[2 * 4 + j];
		int g1 = f[0 * 4 + j] - f[2 * 4 + j];
		int g2 = (f[1 * 4 + j] >> 1) - f[3 * 4 + j];
		int g3 = f[1 * 4 + j] + (f[3 * 4 + j] >> 1);
		int h0 = g0 + g3, h1 = g1 + g2, h2 = g1 - g2, h3 = g0 - g3;
		resid[0 * 4 + j] = (h0 + 32) >> 6;
		resid[1 * 4 + j] = (h1 + 32) >> 6;
		resid[2 * 4 + j] = (h2 + 32) >> 6;
		resid[3 * 4 + j] = (h3 + 32) >> 6;
	}
}

} // namespace wivrn::avc::xform
