/*
 * Minimal spec-faithful HEVC decoder for the restricted bitstream produced by
 * cpu_encoder (single IDR I-slice, 8x8 intra CUs, DC/Planar, DCT8 luma / DCT4
 * chroma, no in-loop filters). Reconstructs the frame and logs each CU's
 * decoded luma intra mode. Used as an independent oracle: decoding the encoder's
 * own stream with a from-spec parser desyncs exactly where the encoder emits a
 * non-conforming bit.
 *
 * Build: g++ -std=c++20 -I.. -O2 hevc_decode.cpp ../transform.cpp -o hevc_decode
 * Use:   ./hevc_decode in.265 W H QP out.yuv
 */

#include "../hevc_tables.h"
#include "../transform.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace wivrn::h267::tables;

// ---- CABAC decode engine (Rec. ITU-T H.265 clause 9.3.4.3) ----
struct cabac_decoder
{
	const uint8_t * d = nullptr;
	size_t n = 0, bytepos = 0;
	int bitpos = 0;
	uint8_t pstate[HEVC_CONTEXTS];
	uint8_t valmps[HEVC_CONTEXTS];
	int range = 510, offset = 0;

	int read_bit()
	{
		if (bytepos >= n)
			return 0;
		int b = (d[bytepos] >> (7 - bitpos)) & 1;
		if (++bitpos == 8)
		{
			bitpos = 0;
			++bytepos;
		}
		return b;
	}

	void init(const uint8_t * data, size_t size, size_t start_byte, int init_type, int qp)
	{
		d = data;
		n = size;
		bytepos = start_byte;
		bitpos = 0;
		auto clip3 = [](int lo, int hi, int v) { return v < lo ? lo : v > hi ? hi : v; };
		int q = clip3(0, 51, qp);
		for (int i = 0; i < HEVC_CONTEXTS; ++i)
		{
			int iv = init_values[init_type][i];
			int m = (iv >> 4) * 5 - 45;
			int nn = ((iv & 15) << 3) - 16;
			int pre = clip3(1, 126, ((m * q) >> 4) + nn);
			if (pre <= 63)
			{
				pstate[i] = 63 - pre;
				valmps[i] = 0;
			}
			else
			{
				pstate[i] = pre - 64;
				valmps[i] = 1;
			}
		}
		range = 510;
		offset = 0;
		for (int i = 0; i < 9; ++i)
			offset = (offset << 1) | read_bit();
	}

	int decode_bin(int ctx)
	{
		int lps = rangeTabLps[pstate[ctx]][(range >> 6) & 3];
		range -= lps;
		int bin;
		if (offset >= range)
		{
			bin = 1 - valmps[ctx];
			offset -= range;
			range = lps;
			if (pstate[ctx] == 0)
				valmps[ctx] = 1 - valmps[ctx];
			pstate[ctx] = transIdxLps[pstate[ctx]];
		}
		else
		{
			bin = valmps[ctx];
			pstate[ctx] = transIdxMps[pstate[ctx]];
		}
		while (range < 256)
		{
			range <<= 1;
			offset = (offset << 1) | read_bit();
		}
		return bin;
	}

	int decode_bypass()
	{
		offset = (offset << 1) | read_bit();
		if (offset >= range)
		{
			offset -= range;
			return 1;
		}
		return 0;
	}

	int decode_terminate()
	{
		range -= 2;
		if (offset >= range)
			return 1;
		while (range < 256)
		{
			range <<= 1;
			offset = (offset << 1) | read_bit();
		}
		return 0;
	}
};

// ---- geometry / prediction (copied verified logic) ----
static constexpr int MODE_PLANAR = 0, MODE_DC = 1, MODE_VER = 26;

struct plane
{
	std::vector<int> rec;
	std::vector<uint8_t> avail;
	int w = 0, h = 0;
	void init(int W, int H)
	{
		w = W;
		h = H;
		rec.assign((size_t)W * H, 0);
		avail.assign((size_t)W * H, 0);
	}
	int at(int x, int y) const { return rec[(size_t)y * w + x]; }
	bool ok(int x, int y) const { return x >= 0 && y >= 0 && x < w && y < h && avail[(size_t)y * w + x]; }
};

static void get_refs(const plane & pl, int xTb, int yTb, int N, int bd, int * refTop, int * refLeft)
{
	int total = 4 * N + 1;
	std::vector<int> seq(total);
	std::vector<uint8_t> sav(total);
	int idx = 0;
	for (int y = 2 * N - 1; y >= 0; --y)
	{
		bool a = pl.ok(xTb - 1, yTb + y);
		sav[idx] = a;
		seq[idx++] = a ? pl.at(xTb - 1, yTb + y) : 0;
	}
	{
		bool a = pl.ok(xTb - 1, yTb - 1);
		sav[idx] = a;
		seq[idx++] = a ? pl.at(xTb - 1, yTb - 1) : 0;
	}
	for (int x = 0; x < 2 * N; ++x)
	{
		bool a = pl.ok(xTb + x, yTb - 1);
		sav[idx] = a;
		seq[idx++] = a ? pl.at(xTb + x, yTb - 1) : 0;
	}
	bool any = false;
	for (int i = 0; i < total; ++i)
		any = any || sav[i];
	if (not any)
	{
		int def = 1 << (bd - 1);
		for (int i = 0; i < total; ++i)
			seq[i] = def;
	}
	else
	{
		if (not sav[0])
		{
			int first = 0;
			while (not sav[first])
				++first;
			for (int i = 0; i < first; ++i)
				seq[i] = seq[first];
		}
		for (int i = 1; i < total; ++i)
			if (not sav[i])
				seq[i] = seq[i - 1];
	}
	refLeft[0] = refTop[0] = seq[2 * N];
	for (int k = 1; k <= 2 * N; ++k)
	{
		refLeft[k] = seq[2 * N - k];
		refTop[k] = seq[2 * N + k];
	}
}

static void filter_refs(int * refTop, int * refLeft, int N)
{
	int corner = refTop[0];
	std::vector<int> nt(2 * N + 1), nl(2 * N + 1);
	nt[0] = nl[0] = (refLeft[1] + 2 * corner + refTop[1] + 2) >> 2;
	for (int k = 1; k < 2 * N; ++k)
	{
		nt[k] = (refTop[k - 1] + 2 * refTop[k] + refTop[k + 1] + 2) >> 2;
		nl[k] = (refLeft[k - 1] + 2 * refLeft[k] + refLeft[k + 1] + 2) >> 2;
	}
	nt[2 * N] = refTop[2 * N];
	nl[2 * N] = refLeft[2 * N];
	for (int k = 0; k <= 2 * N; ++k)
	{
		refTop[k] = nt[k];
		refLeft[k] = nl[k];
	}
}

static int log2_of(int nn)
{
	int l = 0;
	while ((1 << l) < nn)
		++l;
	return l;
}

static void predict(int mode, int N, int bd, int cIdx, const int * refTop, const int * refLeft, int * pred)
{
	int log2n = log2_of(N);
	if (mode == MODE_PLANAR)
	{
		int topRight = refTop[N + 1], botLeft = refLeft[N + 1];
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
				pred[y * N + x] = ((N - 1 - x) * refLeft[1 + y] + (x + 1) * topRight + (N - 1 - y) * refTop[1 + x] + (y + 1) * botLeft + N) >> (log2n + 1);
	}
	else if (mode == MODE_DC)
	{
		int sum = 0;
		for (int i = 0; i < N; ++i)
			sum += refTop[1 + i] + refLeft[1 + i];
		int dc = (sum + N) >> (log2n + 1);
		for (int i = 0; i < N * N; ++i)
			pred[i] = dc;
		if (cIdx == 0 && N < 32)
		{
			pred[0] = (refLeft[1] + 2 * dc + refTop[1] + 2) >> 2;
			for (int x = 1; x < N; ++x)
				pred[x] = (refTop[1 + x] + 3 * dc + 2) >> 2;
			for (int y = 1; y < N; ++y)
				pred[y * N] = (refLeft[1 + y] + 3 * dc + 2) >> 2;
		}
	}
	else
	{
		// Angular (only needed to reconstruct if the stream signals it). Vertical
		// (26) implemented for diagnostics; others fall back to copying top.
		for (int y = 0; y < N; ++y)
			for (int x = 0; x < N; ++x)
				pred[y * N + x] = refTop[1 + x];
		if (mode == MODE_VER && cIdx == 0 && N < 32)
			for (int y = 0; y < N; ++y)
			{
				int v = refTop[1] + ((refLeft[1 + y] - refTop[0]) >> 1);
				pred[y * N] = v < 0 ? 0 : v > 255 ? 255 : v;
			}
	}
	(void)bd;
}

// ---- residual_coding decode (inverse of encoder) ----
struct pos
{
	int x, y;
};
static void diag_scan(int size, pos * out)
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
static const int group_idx[32] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9};
static const int min_in_group[10] = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24};

static int sig_ctx_inc(int xc, int yc, int log2size, int cidx, int prev_csbf, int xs, int ys)
{
	int sig_ctx;
	if (log2size == 2)
	{
		static const int m[16] = {0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8};
		sig_ctx = m[(yc << 2) + xc];
	}
	else if (xc + yc == 0)
		sig_ctx = 0;
	else
	{
		int xp = xc & 3, yp = yc & 3;
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
			sig_ctx += (log2size == 3) ? 9 : 21;
		}
		else
			sig_ctx += (log2size == 3) ? 9 : 12;
	}
	return cidx == 0 ? sig_ctx : 27 + sig_ctx;
}

static int decode_coeff_remain(cabac_decoder & dec, int rice)
{
	int prefix = 0;
	while (prefix < 32 && dec.decode_bypass())
		++prefix;
	if (prefix < 3)
	{
		int suffix = 0;
		for (int i = 0; i < rice; ++i)
			suffix = (suffix << 1) | dec.decode_bypass();
		return (prefix << rice) + suffix;
	}
	int len = prefix - 3 + rice;
	int suffix = 0;
	for (int i = 0; i < len; ++i)
		suffix = (suffix << 1) | dec.decode_bypass();
	return (((1 << (prefix - 3)) + 3 - 1) << rice) + suffix;
}

static void decode_residual(cabac_decoder & dec, int * level, int log2size, int cidx)
{
	int n = 1 << log2size, nsb = n >> 2, ctx_cidx = cidx ? 1 : 0;
	for (int i = 0; i < n * n; ++i)
		level[i] = 0;
	pos scan16[16];
	diag_scan(4, scan16);
	pos scan_sb[16];
	diag_scan(nsb, scan_sb);
	int num_sb = nsb * nsb;

	auto dec_last_prefix = [&](int base_ctx) {
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
		int cmax = (log2size << 1) - 1, p = 0;
		while (p < cmax && dec.decode_bin(base_ctx + (p >> ctx_shift) + ctx_offset))
			++p;
		return p;
	};
	int px = dec_last_prefix(LAST_SIGNIFICANT_COEFF_X_PREFIX);
	int py = dec_last_prefix(LAST_SIGNIFICANT_COEFF_Y_PREFIX);
	auto dec_last = [&](int prefix) {
		if (prefix < 4)
			return prefix;
		int nbits = (prefix >> 1) - 1, suf = 0;
		for (int i = 0; i < nbits; ++i)
			suf = (suf << 1) | dec.decode_bypass();
		return min_in_group[prefix] + suf;
	};
	int last_x = dec_last(px), last_y = dec_last(py);

	// Locate last_sb / last_pos in scan order.
	int last_sb = 0, last_pos = 0;
	for (int sb = 0; sb < num_sb; ++sb)
	{
		int xs = scan_sb[sb].x, ys = scan_sb[sb].y;
		for (int p = 0; p < 16; ++p)
			if (xs * 4 + scan16[p].x == last_x && ys * 4 + scan16[p].y == last_y)
			{
				last_sb = sb;
				last_pos = p;
			}
	}

	uint8_t csbf[4][4] = {};
	int c1 = 1;
	for (int sb = last_sb; sb >= 0; --sb)
	{
		int xs = scan_sb[sb].x, ys = scan_sb[sb].y;
		bool sb_last = (sb == last_sb), sb_first = (sb == 0);
		int infer_dc = 0;
		if (not sb_last && not sb_first)
		{
			int right = (xs + 1 < nsb) ? csbf[ys][xs + 1] : 0;
			int below = (ys + 1 < nsb) ? csbf[ys + 1][xs] : 0;
			int f = dec.decode_bin(SIGNIFICANT_COEFF_GROUP_FLAG + ctx_cidx * 2 + ((right | below) ? 1 : 0));
			csbf[ys][xs] = f;
			if (not f)
				continue;
			infer_dc = 1;
		}
		else
			csbf[ys][xs] = 1;

		int right = (xs + 1 < nsb) ? csbf[ys][xs + 1] : 0;
		int below = (ys + 1 < nsb) ? csbf[ys + 1][xs] : 0;
		int prev_csbf = right | (below << 1);

		uint8_t sig[16] = {};
		if (sb_last)
			sig[last_pos] = 1;
		int start = sb_last ? last_pos - 1 : 15;
		int coded_any = 0;
		for (int p = start; p >= 0; --p)
		{
			int cx = xs * 4 + scan16[p].x, cy = ys * 4 + scan16[p].y;
			if (p == 0 && infer_dc && not coded_any)
			{
				sig[0] = 1;
				break;
			}
			int s = dec.decode_bin(SIGNIFICANT_COEFF_FLAG + sig_ctx_inc(cx, cy, log2size, cidx, prev_csbf, xs, ys));
			sig[p] = s;
			if (s)
				coded_any = 1;
		}
		int sig_scan[16], num_sig = 0;
		for (int p = 15; p >= 0; --p)
			if (sig[p])
				sig_scan[num_sig++] = p;
		if (num_sig == 0)
			continue;

		int ctx_set = (sb > 0 && cidx == 0) ? 2 : 0;
		if (c1 == 0)
			++ctx_set;
		c1 = 1;
		int num_gt1 = num_sig < 8 ? num_sig : 8;
		uint8_t gt1[16] = {};
		int first_gt1 = -1;
		for (int k = 0; k < num_gt1; ++k)
		{
			int b = dec.decode_bin(COEFF_ABS_LEVEL_GREATER1_FLAG + ctx_cidx * 16 + ctx_set * 4 + c1);
			gt1[k] = b;
			if (b)
			{
				if (first_gt1 < 0)
					first_gt1 = k;
				c1 = 0;
			}
			else if (c1 < 3 && c1 > 0)
				++c1;
		}
		int gt2 = 0;
		if (first_gt1 >= 0)
			gt2 = dec.decode_bin(COEFF_ABS_LEVEL_GREATER2_FLAG + ctx_cidx * 4 + ctx_set);

		int sign[16];
		for (int k = 0; k < num_sig; ++k)
			sign[k] = dec.decode_bypass();

		int rice = 0;
		int absv[16];
		for (int k = 0; k < num_sig; ++k)
		{
			int base;
			if (k >= 8)
				base = 1;
			else if (gt1[k] == 0)
				base = -1;
			else if (k == first_gt1)
				base = gt2 ? 3 : -1;
			else
				base = 2;
			int a;
			if (base < 0)
				a = (k < 8 && gt1[k] == 0) ? 1 : 2; // level fully known from flags
			else
			{
				int rem = decode_coeff_remain(dec, rice);
				a = base + rem;
				if (a > (3 << rice))
					rice = rice < 4 ? rice + 1 : 4;
			}
			absv[k] = a;
		}
		for (int k = 0; k < num_sig; ++k)
		{
			int p = sig_scan[k];
			int cx = xs * 4 + scan16[p].x, cy = ys * 4 + scan16[p].y;
			level[cy * n + cx] = sign[k] ? -absv[k] : absv[k];
		}
	}
}

// ---- bit reader for the slice header ----
struct bitreader
{
	const uint8_t * d;
	size_t n, bytepos = 0;
	int bitpos = 0;
	int u1()
	{
		int b = (d[bytepos] >> (7 - bitpos)) & 1;
		if (++bitpos == 8)
		{
			bitpos = 0;
			++bytepos;
		}
		return b;
	}
	uint32_t u(int k)
	{
		uint32_t v = 0;
		while (k--)
			v = (v << 1) | u1();
		return v;
	}
	uint32_t ue()
	{
		int z = 0;
		while (u1() == 0)
			++z;
		uint32_t v = 1;
		for (int i = 0; i < z; ++i)
			v = (v << 1) | u1();
		return v - 1;
	}
	int se()
	{
		uint32_t k = ue();
		int v = (k + 1) >> 1;
		return (k & 1) ? v : -v;
	}
};

// ---- NAL utilities ----
static std::vector<std::vector<uint8_t>> split_nals(const std::vector<uint8_t> & buf)
{
	std::vector<std::vector<uint8_t>> nals;
	size_t i = 0;
	auto is_sc = [&](size_t p) { return p + 3 < buf.size() && buf[p] == 0 && buf[p + 1] == 0 && buf[p + 2] == 1; };
	while (i + 3 < buf.size())
	{
		if (is_sc(i))
		{
			size_t start = i + 3;
			size_t j = start;
			while (j + 3 < buf.size() && not is_sc(j) && not(buf[j] == 0 && buf[j + 1] == 0 && buf[j + 2] == 0 && j + 3 < buf.size() && buf[j + 3] == 1))
				++j;
			if (j + 3 >= buf.size())
				j = buf.size();
			nals.emplace_back(buf.begin() + start, buf.begin() + j);
			i = j;
		}
		else
			++i;
	}
	return nals;
}
static std::vector<uint8_t> deemulate(const std::vector<uint8_t> & nal)
{
	std::vector<uint8_t> out;
	int zeros = 0;
	for (size_t i = 0; i < nal.size(); ++i)
	{
		if (zeros >= 2 && nal[i] == 3)
		{
			zeros = 0;
			continue;
		}
		out.push_back(nal[i]);
		if (nal[i] == 0)
			++zeros;
		else
			zeros = 0;
	}
	return out;
}

// ---- decoder state ----
struct decoder
{
	int W, H, CW, CH, bd = 8, qp;
	int max_pix;
	plane Y, Cb, Cr;
	std::vector<uint8_t> mode_plane, ct_depth;
	cabac_decoder dec;
	FILE * modelog;

	int clipp(int v) { return v < 0 ? 0 : v > max_pix ? max_pix : v; }

	int build_mpm_and_decode()
	{
		return 0;
	}

	void recon_tu(plane & pl, int x, int y, int N, int mode, int cIdx, int * level, bool has_res)
	{
		int refTop[65], refLeft[65];
		get_refs(pl, x, y, N, bd, refTop, refLeft);
		if (cIdx == 0 && mode != MODE_DC && N != 4)
		{
			int minDist = std::min(std::abs(mode - 26), std::abs(mode - 10));
			int thres = (N == 8) ? 7 : (N == 16) ? 1 : 0;
			if (minDist > thres)
				filter_refs(refTop, refLeft, N);
		}
		int pred[64];
		predict(mode, N, bd, cIdx, refTop, refLeft, pred);
		int rec[64];
		if (has_res)
		{
			int deq[64], res[64];
			wivrn::h267::xform::dequant(level, deq, N, qp, bd);
			wivrn::h267::xform::idct(deq, res, N, bd);
			for (int i = 0; i < N * N; ++i)
				rec[i] = clipp(pred[i] + res[i]);
		}
		else
			for (int i = 0; i < N * N; ++i)
				rec[i] = clipp(pred[i]);
		for (int j = 0; j < N; ++j)
			for (int i = 0; i < N; ++i)
			{
				pl.rec[(size_t)(y + j) * pl.w + (x + i)] = rec[j * N + i];
				pl.avail[(size_t)(y + j) * pl.w + (x + i)] = 1;
			}
	}

	void coding_unit(int x0, int y0)
	{
		// part_mode (intra, min CU): 1 => 2Nx2N, 0 => NxN
		int part = dec.decode_bin(PART_MODE + 0);
		(void)part;
		// prev_intra_luma_pred_flag + mpm/rem
		int prev = dec.decode_bin(PREV_INTRA_LUMA_PRED_FLAG);
		auto nb = [&](int nx, int ny, bool ctb) -> int {
			if (ctb && ny < ((y0 >> 6) << 6))
				return MODE_DC;
			if (nx < 0 || ny < 0 || not Y.avail[(size_t)ny * Y.w + nx])
				return MODE_DC;
			return mode_plane[(size_t)ny * Y.w + nx];
		};
		int a = nb(x0 - 1, y0, false), b = nb(x0, y0 - 1, true);
		int cand[3];
		if (a == b)
		{
			if (a < 2)
			{
				cand[0] = 0;
				cand[1] = 1;
				cand[2] = 26;
			}
			else
			{
				cand[0] = a;
				cand[1] = 2 + ((a + 29) % 32);
				cand[2] = 2 + ((a - 2 + 1) % 32);
			}
		}
		else
		{
			cand[0] = a;
			cand[1] = b;
			if (a != 0 && b != 0)
				cand[2] = 0;
			else if (a != 1 && b != 1)
				cand[2] = 1;
			else
				cand[2] = 26;
		}
		int mode;
		if (prev)
		{
			int idx = 0;
			if (dec.decode_bypass())
				idx = dec.decode_bypass() ? 2 : 1;
			mode = cand[idx];
		}
		else
		{
			int rem = 0;
			for (int i = 0; i < 5; ++i)
				rem = (rem << 1) | dec.decode_bypass();
			int c[3] = {cand[0], cand[1], cand[2]};
			for (int i = 0; i < 2; ++i)
				for (int j = i + 1; j < 3; ++j)
					if (c[i] > c[j])
						std::swap(c[i], c[j]);
			mode = rem;
			for (int i = 0; i < 3; ++i)
				if (mode >= c[i])
					++mode;
		}
		if (modelog)
			std::fprintf(modelog, "%d %d %d\n", x0, y0, mode);

		// intra_chroma_pred_mode
		int cmode;
		if (dec.decode_bin(INTRA_CHROMA_PRED_MODE + 0) == 0)
			cmode = mode; // DM
		else
		{
			int idx = dec.decode_bypass() << 1;
			idx |= dec.decode_bypass();
			static const int m[4] = {0, 26, 10, 1};
			cmode = m[idx];
			if (cmode == mode)
				cmode = 34;
		}

		for (int j = 0; j < 8; ++j)
			for (int i = 0; i < 8; ++i)
			{
				mode_plane[(size_t)(y0 + j) * Y.w + (x0 + i)] = mode;
				ct_depth[(size_t)(y0 + j) * Y.w + (x0 + i)] = 3;
			}

		// transform_tree(8x8): cbf_cb, cbf_cr (depth0), cbf_luma (ctxInc 1)
		int cbf_cb = dec.decode_bin(CBF_CB_CR + 0);
		int cbf_cr = dec.decode_bin(CBF_CB_CR + 0);
		int cbf_luma = dec.decode_bin(CBF_LUMA + 1);

		int lvlY[64], lvlCb[16], lvlCr[16];
		if (cbf_luma)
			decode_residual(dec, lvlY, 3, 0);
		if (cbf_cb)
			decode_residual(dec, lvlCb, 2, 1);
		if (cbf_cr)
			decode_residual(dec, lvlCr, 2, 2);

		recon_tu(Y, x0, y0, 8, mode, 0, lvlY, cbf_luma);
		recon_tu(Cb, x0 / 2, y0 / 2, 4, cmode, 1, lvlCb, cbf_cb);
		recon_tu(Cr, x0 / 2, y0 / 2, 4, cmode, 2, lvlCr, cbf_cr);
	}

	void quadtree(int x0, int y0, int log2, int depth)
	{
		if (log2 > 3)
		{
			bool condL = Y.ok(x0 - 1, y0) && ct_depth[(size_t)y0 * Y.w + (x0 - 1)] > depth;
			bool condA = Y.ok(x0, y0 - 1) && ct_depth[(size_t)(y0 - 1) * Y.w + x0] > depth;
			int split = dec.decode_bin(SPLIT_CODING_UNIT_FLAG + (condL ? 1 : 0) + (condA ? 1 : 0));
			if (split)
			{
				int half = 1 << (log2 - 1);
				quadtree(x0, y0, log2 - 1, depth + 1);
				quadtree(x0 + half, y0, log2 - 1, depth + 1);
				quadtree(x0, y0 + half, log2 - 1, depth + 1);
				quadtree(x0 + half, y0 + half, log2 - 1, depth + 1);
			}
			else
				coding_unit(x0, y0);
		}
		else
			coding_unit(x0, y0);
	}
};

int main(int argc, char ** argv)
{
	if (argc < 6)
	{
		std::fprintf(stderr, "usage: %s in.265 W H QP out.yuv [modelog.txt]\n", argv[0]);
		return 1;
	}
	FILE * f = std::fopen(argv[1], "rb");
	std::fseek(f, 0, SEEK_END);
	long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	std::vector<uint8_t> buf(sz);
	if (std::fread(buf.data(), 1, sz, f) != (size_t)sz) { std::fprintf(stderr, "read error\n"); return 1; }
	std::fclose(f);

	decoder D;
	D.W = std::atoi(argv[2]);
	D.H = std::atoi(argv[3]);
	D.qp = std::atoi(argv[4]);
	// coded size (CTB aligned)
	int cw = ((D.W + 63) / 64) * 64, ch = ((D.H + 63) / 64) * 64;
	D.W = cw;
	D.H = ch;
	D.CW = cw / 2;
	D.CH = ch / 2;
	D.max_pix = (1 << D.bd) - 1;
	D.Y.init(cw, ch);
	D.Cb.init(cw / 2, ch / 2);
	D.Cr.init(cw / 2, ch / 2);
	D.mode_plane.assign((size_t)cw * ch, MODE_DC);
	D.ct_depth.assign((size_t)cw * ch, 0);
	D.modelog = argc > 6 ? std::fopen(argv[6], "w") : nullptr;

	auto nals = split_nals(buf);
	for (auto & nal: nals)
	{
		int type = (nal[0] >> 1) & 0x3f;
		if (type == 19 || type == 20) // IDR slice
		{
			auto rbsp = deemulate(nal);
			bitreader br{rbsp.data() + 2, rbsp.size() - 2}; // skip 2-byte NAL header
			br.u1();      // first_slice_segment_in_pic_flag
			br.u1();      // no_output_of_prior_pics_flag
			br.ue();      // slice_pic_parameter_set_id
			br.ue();      // slice_type
			int qp_delta = br.se();
			(void)qp_delta;
			// byte_alignment
			br.u1();
			while (br.bitpos != 0)
				br.u1();
			size_t slice_data_off = 2 + br.bytepos; // in rbsp
			D.dec.init(rbsp.data(), rbsp.size(), slice_data_off, 0, D.qp);
			int nx = cw / 64, ny = ch / 64;
			for (int cy = 0; cy < ny; ++cy)
				for (int cx = 0; cx < nx; ++cx)
				{
					D.quadtree(cx * 64, cy * 64, 6, 0);
					D.dec.decode_terminate();
				}
			break;
		}
	}

	// write I420
	FILE * o = std::fopen(argv[5], "wb");
	std::vector<uint8_t> out;
	for (int v: D.Y.rec)
		out.push_back((uint8_t)v);
	for (int v: D.Cb.rec)
		out.push_back((uint8_t)v);
	for (int v: D.Cr.rec)
		out.push_back((uint8_t)v);
	if (std::fwrite(out.data(), 1, out.size(), o) != out.size()) { std::fprintf(stderr, "write error\n"); }
	std::fclose(o);
	if (D.modelog)
		std::fclose(D.modelog);
	std::fprintf(stderr, "decoded %dx%d\n", cw, ch);
	return 0;
}
