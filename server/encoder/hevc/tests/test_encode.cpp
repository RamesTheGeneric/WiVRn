/*
 * Encode a synthetic structured image with the CPU reference intra encoder and
 * dump: the Annex-B .265 stream, the encoder's own reconstruction (I420), and
 * the source (I420). An external decoder must reproduce the reconstruction
 * exactly; PSNR(recon, source) measures quality.
 *
 * Build: g++ -std=c++20 -I.. test_encode.cpp ../cpu_encoder.cpp ../transform.cpp \
 *          ../residual_coding.cpp ../cabac_reference.cpp ../param_sets.cpp ../bitwriter.cpp -o test_encode
 * Use:   ./test_encode 256 256 26 out   # writes out.265 out_recon.yuv out_src.yuv
 */

#include "../cpu_encoder.h"
#include "../param_sets.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace wivrn::hevc;

static void write_file(const std::string & path, const void * data, size_t n)
{
	FILE * f = std::fopen(path.c_str(), "wb");
	std::fwrite(data, 1, n, f);
	std::fclose(f);
}

int main(int argc, char ** argv)
{
	hevc_config cfg;
	cfg.width = argc > 1 ? std::atoi(argv[1]) : 256;
	cfg.height = argc > 2 ? std::atoi(argv[2]) : 256;
	cfg.qp = argc > 3 ? std::atoi(argv[3]) : 26;
	cfg.bit_depth = 8;
	cfg.max_tb_log2_size = 3; // this encoder uses 8x8 luma / 4x4 chroma TUs
	const std::string base = argc > 4 ? argv[4] : "out";

	const int W = cfg.coded_width(), H = cfg.coded_height();
	const int CW = W / 2, CH = H / 2;

	// Synthetic content: luma gradient + sinusoid + a bright rectangle; chroma
	// smooth gradients. Produces non-trivial residuals across the frame.
	yuv_image img;
	img.width = W;
	img.height = H;
	img.Y.resize((size_t)W * H);
	img.Cb.resize((size_t)CW * CH);
	img.Cr.resize((size_t)CW * CH);
	for (int y = 0; y < H; ++y)
		for (int x = 0; x < W; ++x)
		{
			double v = 128 + 90.0 * std::sin(x * 0.06) * std::cos(y * 0.045) + (x - W / 2) * 0.15;
			if (x > W / 4 && x < W / 2 && y > H / 3 && y < 2 * H / 3)
				v = 235; // flat bright rectangle
			int iv = (int)std::lround(v);
			img.Y[(size_t)y * W + x] = (uint16_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
		}
	for (int y = 0; y < CH; ++y)
		for (int x = 0; x < CW; ++x)
		{
			img.Cb[(size_t)y * CW + x] = (uint16_t)(128 + (x - CW / 2) / 3);
			img.Cr[(size_t)y * CW + x] = (uint16_t)(128 + (y - CH / 2) / 3);
		}

	yuv_image recon;
	auto ps = build_parameter_sets(cfg);
	auto frame = encode_intra_frame(cfg, img, &recon);

	std::vector<uint8_t> stream = ps;
	stream.insert(stream.end(), frame.begin(), frame.end());
	write_file(base + ".265", stream.data(), stream.size());

	// Dump source and reconstruction as I420 (8-bit) for external comparison.
	auto dump_i420 = [&](const std::string & path, const yuv_image & im) {
		std::vector<uint8_t> buf;
		for (auto v: im.Y)
			buf.push_back((uint8_t)v);
		for (auto v: im.Cb)
			buf.push_back((uint8_t)v);
		for (auto v: im.Cr)
			buf.push_back((uint8_t)v);
		write_file(path, buf.data(), buf.size());
	};
	dump_i420(base + "_src.yuv", img);
	dump_i420(base + "_recon.yuv", recon);

	// In-process PSNR of reconstruction vs source (encoder quality).
	double mse = 0;
	for (size_t i = 0; i < img.Y.size(); ++i)
	{
		double d = (double)recon.Y[i] - img.Y[i];
		mse += d * d;
	}
	mse /= img.Y.size();
	double psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99.0;

	std::fprintf(stderr,
	             "coded %dx%d qp %d  stream %zu bytes (%.3f bpp luma)  luma PSNR(recon,src) %.2f dB\n",
	             W, H, cfg.qp, stream.size(), stream.size() * 8.0 / (W * H), psnr);
	return 0;
}
