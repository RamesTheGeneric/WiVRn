// Offline soak + contention harness — reproduce the two-eye, sustained,
// contended conditions that a single-shot offline test misses (the scoreboard
// recon passed single-shot but raced under live GPU contention).
//
// Two encoder instances (each its own device) run concurrently on the one
// physical GPU, so their dispatches contend for CUs exactly like the two eyes
// do live. Each runs a loop of *moving* synthetic frames and, every frame,
// checks the GPU reconstruction bit-exact against the CPU reference — a race
// shows up as a nonzero mismatch. Per-eye timing percentiles are reported.
//
// Build/run (from server/encoder/h264/):
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_recon_dc.comp   -o /tmp/r264.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_cavlc_emit.comp -o /tmp/emit264.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_cavlc_prefix.comp -o /tmp/pfx264.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_cavlc_stitch.comp -o /tmp/stitch264.spv
//   g++ -std=c++20 -O2 -I. -I.. -I../hevc/gpu gpu/tests/soak264.cpp gpu/encoder264.cpp \
//       cpu_encoder264.cpp cavlc.cpp transform264.cpp param_sets264.cpp ../hevc/bitwriter.cpp \
//       -lvulkan -pthread -o /tmp/soak264
//   /tmp/soak264 /tmp/r264.spv /tmp/emit264.spv /tmp/pfx264.spv /tmp/stitch264.spv [frames]
#include "../encoder264.h"
#include "../../cpu_encoder264.h"
#include "../../transform264.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace wivrn::avc;

namespace
{
const char * g_spv[4];
int g_frames = 300;
int g_w = 768, g_h = 832, g_qp = 26;

// Deterministic content that shifts with frame `f` (simulated motion — the
// condition under which the scoreboard's stale-neighbour reads became visible).
void fill_content(int f, int cw, int ch, yuv_image & img, std::vector<uint8_t> & luma, std::vector<uint8_t> & chroma)
{
	const int cw2 = cw / 2, ch2 = ch / 2;
	img.width = cw; img.height = ch;
	img.Y.resize((size_t)cw * ch); img.Cb.resize((size_t)cw2 * ch2); img.Cr.resize((size_t)cw2 * ch2);
	for (int y = 0; y < ch; ++y)
		for (int x = 0; x < cw; ++x)
			img.Y[(size_t)y * cw + x] = (uint8_t)(20 + ((x + f * 3) * 180 / cw) + (((x * 7 + y * 13 + f * 5)) & 63));
	for (int y = 0; y < ch2; ++y)
		for (int x = 0; x < cw2; ++x) {
			img.Cb[(size_t)y * cw2 + x] = (uint8_t)(80 + ((x + f) * 90 / cw2) + ((x ^ y) & 31));
			img.Cr[(size_t)y * cw2 + x] = (uint8_t)(180 - ((y + f * 2) * 90 / ch2) + ((x * 3) & 15));
		}
	// GPU inputs: luma extent-packed, chroma interleaved CbCr.
	luma = img.Y;
	chroma.resize((size_t)cw * ch / 2);
	for (int y = 0; y < ch2; ++y)
		for (int x = 0; x < cw2; ++x) {
			chroma[((size_t)y * cw2 + x) * 2 + 0] = img.Cb[(size_t)y * cw2 + x];
			chroma[((size_t)y * cw2 + x) * 2 + 1] = img.Cr[(size_t)y * cw2 + x];
		}
}

struct result
{
	int mismatches = 0;
	int first_bad_frame = -1;
	std::vector<double> recon_us, total_us;
};

void run_eye(int eye, result & out)
{
	gpu::encoder enc;
	enc.init_own(g_spv[0], g_spv[1], g_spv[2], g_spv[3]);
	const int cw = ((g_w + 15) / 16) * 16, ch = ((g_h + 15) / 16) * 16, cw2 = cw / 2, ch2 = ch / 2;
	h264_config cfg; cfg.width = g_w; cfg.height = g_h; cfg.qp = g_qp;

	yuv_image img;
	std::vector<uint8_t> luma, chroma;
	std::vector<uint8_t> gRecY(cw * ch), gRecCb(cw2 * ch2), gRecCr(cw2 * ch2);
	std::vector<uint8_t> cRecY, cRecCb, cRecCr;

	for (int f = 0; f < g_frames; ++f)
	{
		// vary content per eye and per frame so the two threads aren't lockstep
		fill_content(f * 2 + eye, cw, ch, img, luma, chroma);
		encode_idr_frame(cfg, img, &cRecY, &cRecCb, &cRecCr); // CPU reference

		enc.encode_frame(cfg, cw, ch, luma.data(), chroma.data(), gRecY.data(), gRecCb.data(), gRecCr.data());

		int mism = 0;
		for (size_t i = 0; i < (size_t)cw * ch; ++i) mism += (gRecY[i] != cRecY[i]);
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) mism += (gRecCb[i] != cRecCb[i]);
		for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) mism += (gRecCr[i] != cRecCr[i]);
		if (mism) { out.mismatches += mism; if (out.first_bad_frame < 0) out.first_bad_frame = f; }

		out.recon_us.push_back(enc.last_timings.recon_us);
		out.total_us.push_back(enc.last_timings.upload + enc.last_timings.recon_us +
		                       enc.last_timings.cavlc_us + enc.last_timings.assemble_us);
	}
}

void stats(const char * tag, std::vector<double> v)
{
	if (v.empty()) return;
	std::sort(v.begin(), v.end());
	size_t n = v.size();
	printf("  %-8s min=%.0f  median=%.0f  p90=%.0f  p99=%.0f  max=%.0f us\n",
	       tag, v.front(), v[n / 2], v[n * 9 / 10], v[n * 99 / 100], v.back());
}
} // namespace

int main(int argc, char ** argv)
{
	if (argc < 5) { printf("usage: %s recon.spv emit.spv prefix.spv stitch.spv [frames]\n", argv[0]); return 2; }
	for (int i = 0; i < 4; ++i) g_spv[i] = argv[i + 1];
	if (argc > 5) g_frames = atoi(argv[5]);
	int eyes = getenv("SOLO") ? 1 : 2;
	if (const char * e = getenv("EYES")) { int n = atoi(e); if (n > 0) eyes = n; }

	printf("soak: %dx%d qp%d, %d frames/eye, %d concurrent eyes (contending)\n", g_w, g_h, g_qp, g_frames, eyes);

	std::vector<result> r(eyes);
	std::vector<std::thread> th;
	for (int i = 0; i < eyes; ++i)
		th.emplace_back(run_eye, i, std::ref(r[i]));
	for (auto & t : th) t.join();

	int total_mism = 0;
	for (int i = 0; i < eyes; ++i) {
		total_mism += r[i].mismatches;
		printf("\neye%d: mismatches=%d%s\n", i, r[i].mismatches,
		       r[i].first_bad_frame >= 0 ? (" (first bad frame " + std::to_string(r[i].first_bad_frame) + ")").c_str() : "");
		stats("recon", r[i].recon_us);
		stats("total", r[i].total_us);
	}

	printf("\n%s (%d total sample mismatches over %d encodes)\n",
	       total_mism == 0 ? "SOAK CLEAN — bit-exact under contention" : "SOAK FAILED — race/corruption detected",
	       total_mism, eyes * g_frames);
	return total_mism ? 1 : 0;
}
