// H.264 CPU encoder conformance: encode a synthetic frame, decode with ffmpeg,
// and assert the decoded YUV equals the encoder's own reconstruction (i.e. a
// conforming decoder reproduces our recon exactly).
//
// Build/run (from server/encoder/h264/):
//   g++ -std=c++20 -I. -I.. tests/test_h264.cpp cpu_encoder264.cpp cavlc.cpp \
//       transform264.cpp param_sets264.cpp ../hevc/bitwriter.cpp -o /tmp/test_h264
//   /tmp/test_h264
#include "../cpu_encoder264.h"
#include "../param_sets264.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace wivrn::avc;

int main()
{
	struct tc
	{
		int w, h, qp;
	};
	const tc cases[] = {{128, 128, 26}, {176, 144, 20}, {256, 160, 32}, {320, 240, 12}, {384, 256, 40}};

	int fails = 0;
	for (auto t: cases)
	{
		h264_config cfg;
		cfg.width = t.w;
		cfg.height = t.h;
		cfg.qp = t.qp;
		const int cw = (int)cfg.coded_width(), ch = (int)cfg.coded_height();
		const int cw2 = cw / 2, ch2 = ch / 2;

		yuv_image img;
		img.width = cw;
		img.height = ch;
		img.Y.resize((size_t)cw * ch);
		img.Cb.resize((size_t)cw2 * ch2);
		img.Cr.resize((size_t)cw2 * ch2);
		for (int y = 0; y < ch; ++y)
			for (int x = 0; x < cw; ++x)
				img.Y[(size_t)y * cw + x] = (uint8_t)(20 + (x * 200 / cw) + ((x ^ y) & 31) + (y / 11) * 3);
		for (int y = 0; y < ch2; ++y)
			for (int x = 0; x < cw2; ++x)
			{
				img.Cb[(size_t)y * cw2 + x] = (uint8_t)(90 + (x * 70 / cw2) + ((x + y) & 15));
				img.Cr[(size_t)y * cw2 + x] = (uint8_t)(170 - (y * 70 / ch2) + ((x * y) & 7));
			}

		std::vector<uint8_t> recY, recCb, recCr;
		auto frame = encode_idr_frame(cfg, img, &recY, &recCb, &recCr);

		std::string base = "/tmp/h264_" + std::to_string(t.w) + "x" + std::to_string(t.h);
		std::string h264 = base + ".264", yuv = base + ".yuv";
		FILE * f = fopen(h264.c_str(), "wb");
		fwrite(frame.data(), 1, frame.size(), f);
		fclose(f);
		std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i " + h264 + " -f rawvideo -pix_fmt yuv420p " + yuv;
		int rc = system(cmd.c_str());

		int mism = 0;
		bool ok = (rc == 0);
		if (ok)
		{
			FILE * yf = fopen(yuv.c_str(), "rb");
			// ffmpeg outputs display size (== coded here, all 16-multiples)
			std::vector<uint8_t> dec((size_t)cw * ch * 3 / 2);
			size_t got = yf ? fread(dec.data(), 1, dec.size(), yf) : 0;
			if (yf)
				fclose(yf);
			if (got != dec.size())
				ok = false;
			else
			{
				const uint8_t * dY = dec.data();
				const uint8_t * dCb = dY + (size_t)cw * ch;
				const uint8_t * dCr = dCb + (size_t)cw2 * ch2;
				for (size_t i = 0; i < (size_t)cw * ch; ++i)
					if (dY[i] != recY[i])
						++mism;
				for (size_t i = 0; i < (size_t)cw2 * ch2; ++i)
					if (dCb[i] != recCb[i])
						++mism;
				for (size_t i = 0; i < (size_t)cw2 * ch2; ++i)
					if (dCr[i] != recCr[i])
						++mism;
			}
		}
		bool pass = ok && mism == 0;
		if (not pass)
			++fails;
		printf("%dx%d qp%d: frame=%zu bytes, decode %s, mismatch=%d  %s\n",
		       t.w, t.h, t.qp, frame.size(), ok ? "ok" : "FAIL", mism, pass ? "PASS" : "FAIL");
	}
	printf("\n%s (%d failures)\n", fails == 0 ? "ALL CONFORMANT" : "FAILURES", fails);
	return fails == 0 ? 0 : 1;
}
