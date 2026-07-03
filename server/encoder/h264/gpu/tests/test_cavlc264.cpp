// M3 validation: GPU recon -> GPU parallel CAVLC emit -> (CPU stitch) assembled
// frame must be byte-exact with the CPU encoder, and decode pixel-exact via
// ffmpeg. The stitch is done on the CPU here (trivial concatenation); the GPU
// stitch is a separate perf step.
//
// Build/run (from server/encoder/h264/):
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_recon_dc.comp -o /tmp/r264.spv
//   glslangValidator -V --target-env vulkan1.3 gpu/shaders/h264_cavlc_emit.comp -o /tmp/emit264.spv
//   g++ -std=c++20 -I. -I.. gpu/tests/test_cavlc264.cpp cpu_encoder264.cpp cavlc.cpp \
//       transform264.cpp param_sets264.cpp ../hevc/bitwriter.cpp -lvulkan -o /tmp/test_cavlc264
//   /tmp/test_cavlc264 /tmp/r264.spv /tmp/emit264.spv
#include "../../../hevc/gpu/vk_compute.h"
#include "../../cpu_encoder264.h"
#include "../../param_sets264.h"
#include "../../transform264.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace wivrn::avc;
using namespace wivrn::h267::gpu;
using bw = wivrn::h267::bitwriter;

struct RPC { uint32_t mbw, mbh; int32_t qp, qpc; uint32_t cw, ch, ew, eh, nmb; };
struct EPC { uint32_t mbw, mbh, stride_words, lgw, cgw, cgh; };

struct PPC { uint32_t nmb, base_bits; };
struct SPC { uint32_t nmb, stride_words; };

int main(int argc, char ** argv)
{
	const char * rspv = argv[1], *espv = argv[2];
	const char * pspv = argc > 3 ? argv[3] : nullptr, *sspv = argc > 4 ? argv[4] : nullptr;
	vk_compute vk;
	vk.init();
	auto rpipe = vk.make_pipeline(rspv, 14, sizeof(RPC));
	auto epipe = vk.make_pipeline(espv, 8, sizeof(EPC));
	bool haveStitch = pspv && sspv;
	vk_compute::pipeline ppipe{}, spipe{};
	if (haveStitch) { ppipe = vk.make_pipeline(pspv, 3, sizeof(PPC)); spipe = vk.make_pipeline(sspv, 4, sizeof(SPC)); }
	const uint32_t STRIDE = 256; // words per MB scratch (8192 bits)

	int fails = 0;
	struct tc { int w, h, qp; };
	for (auto t : {tc{64, 48, 26}, tc{128, 96, 20}, tc{256, 160, 34}, tc{768, 832, 26}})
	{
		h264_config cfg; cfg.width = t.w; cfg.height = t.h; cfg.qp = t.qp;
		const int cw = cfg.coded_width(), ch = cfg.coded_height(), cw2 = cw / 2, ch2 = ch / 2;
		const int ew = t.w, eh = t.h, mbw = cfg.mb_width(), mbh = cfg.mb_height(), nmb = mbw * mbh;

		yuv_image img; img.width = cw; img.height = ch;
		img.Y.resize((size_t)cw * ch); img.Cb.resize((size_t)cw2 * ch2); img.Cr.resize((size_t)cw2 * ch2);
		for (int y = 0; y < ch; ++y) for (int x = 0; x < cw; ++x)
			img.Y[(size_t)y * cw + x] = (uint8_t)(20 + (x * 180 / cw) + ((x * 7 + y * 13) & 63) + (y / 9) * 2);
		for (int y = 0; y < ch2; ++y) for (int x = 0; x < cw2; ++x) {
			img.Cb[(size_t)y * cw2 + x] = (uint8_t)(80 + (x * 90 / cw2) + ((x ^ y) & 31));
			img.Cr[(size_t)y * cw2 + x] = (uint8_t)(180 - (y * 90 / ch2) + ((x * 3) & 15));
		}
		auto frameCPU = encode_idr_frame(cfg, img);

		auto round4 = [](size_t n) { return (n + 3) & ~size_t(3); };
		auto bSY = vk.make_buffer(round4((size_t)ew * eh));
		auto bSC = vk.make_buffer(round4((size_t)ew * eh / 2));
		auto bRY = vk.make_buffer((size_t)cw * ch * 4, true);
		auto bRCb = vk.make_buffer((size_t)cw2 * ch2 * 4, true);
		auto bRCr = vk.make_buffer((size_t)cw2 * ch2 * 4, true);
		auto bLDC = vk.make_buffer((size_t)nmb * 16 * 2, true);  // int16 levels
		auto bLAC = vk.make_buffer((size_t)nmb * 256 * 2, true); // int16 levels
		auto bCDC = vk.make_buffer((size_t)nmb * 8 * 2, true);   // int16 levels
		auto bCAC = vk.make_buffer((size_t)nmb * 128 * 2, true); // int16 levels
		auto bNL = vk.make_buffer((size_t)(cw / 4) * (ch / 4) * 4, true);
		auto bNC = vk.make_buffer((size_t)2 * (cw / 8) * (ch / 8) * 4, true);
		auto bScratch = vk.make_buffer((size_t)nmb * STRIDE * 4, true);
		auto bBitLen = vk.make_buffer((size_t)nmb * 4, true);

		memcpy(bSY.ptr, img.Y.data(), (size_t)ew * eh);
		{ uint8_t * c = (uint8_t *)bSC.ptr; for (int y = 0; y < ch2; ++y) for (int x = 0; x < cw2; ++x) {
			c[((size_t)y * cw2 + x) * 2 + 0] = img.Cb[(size_t)y * cw2 + x];
			c[((size_t)y * cw2 + x) * 2 + 1] = img.Cr[(size_t)y * cw2 + x]; } }

		// recon (scoreboard: single dispatch of persistent workgroups)
		const int qpc = wivrn::avc::xform::chroma_qp(t.qp);
		auto bOrd = vk.make_buffer((size_t)nmb * 4);
		auto bClaim = vk.make_buffer(4, true);
		auto bDone = vk.make_buffer((size_t)nmb * 4, true);
		{
			auto * ord = (uint32_t *)bOrd.ptr; uint32_t k = 0;
			for (int d = 0; d <= mbw + mbh - 2; ++d)
				for (int mbx = std::max(0, d - (mbh - 1)); mbx <= std::min(d, mbw - 1); ++mbx)
					ord[k++] = uint32_t((d - mbx) * mbw + mbx);
			memset(bClaim.ptr, 0, 4); memset(bDone.ptr, 0, (size_t)nmb * 4);
		}
		std::vector<vk_compute::buffer *> rbind = {&bSY, &bSC, &bRY, &bRCb, &bRCr, &bLDC, &bLAC, &bCDC, &bCAC, &bNL, &bNC, &bOrd, &bClaim, &bDone};
		RPC rpc{(uint32_t)mbw, (uint32_t)mbh, t.qp, qpc, (uint32_t)cw, (uint32_t)ch, (uint32_t)ew, (uint32_t)eh, (uint32_t)nmb};
		vk.run(rpipe, rbind, std::min(nmb, 64), 1, 1, &rpc, sizeof(rpc));

		// emit
		EPC epc{(uint32_t)mbw, (uint32_t)mbh, STRIDE, (uint32_t)(cw / 4), (uint32_t)(cw / 8), (uint32_t)(ch / 8)};
		std::vector<vk_compute::buffer *> ebind = {&bLDC, &bLAC, &bCDC, &bCAC, &bNL, &bNC, &bScratch, &bBitLen};
		vk.run(epipe, ebind, (uint32_t)nmb, 1, 1, &epc, sizeof(epc)); // one workgroup (32 lanes) per MB

		// Optional GPU stitch: prefix-sum + parallel bit-concatenation, compared to
		// a CPU concatenation of the same per-MB bits.
		bool stitchOK = true;
		if (haveStitch) {
			auto bOff = vk.make_buffer((size_t)nmb * 4, true);
			auto bTot = vk.make_buffer(4, true);
			auto bOut = vk.make_buffer((size_t)nmb * STRIDE * 4, true); // generous, zeroed
			PPC ppc{(uint32_t)nmb, 0u}; // base_bits=0: CPU concat below also starts at bit 0
			vk.run(ppipe, {&bBitLen, &bOff, &bTot}, 1, 1, 1, &ppc, sizeof(ppc));
			SPC spc{(uint32_t)nmb, STRIDE};
			vk.run(spipe, {&bScratch, &bBitLen, &bOff, &bOut}, (uint32_t)((nmb + 63) / 64), 1, 1, &spc, sizeof(spc));
			uint32_t totbits = ((uint32_t *)bTot.ptr)[0];
			// CPU concat of MB bits
			bw wc;
			const uint32_t * scr2 = (const uint32_t *)bScratch.ptr;
			const uint32_t * bl2 = (const uint32_t *)bBitLen.ptr;
			for (int mb = 0; mb < nmb; ++mb) {
				uint32_t L = bl2[mb]; const uint32_t * base = scr2 + (size_t)mb * STRIDE;
				for (uint32_t i = 0; i < L; ++i) wc.put_bit((base[i >> 5] >> (31 - (i & 31))) & 1u);
			}
			auto ref = wc.bytes();
			const uint32_t * ow = (const uint32_t *)bOut.ptr;
			size_t nbytes = (totbits + 7) / 8;
			stitchOK = (nbytes == ref.size());
			for (size_t i = 0; stitchOK && i < nbytes; ++i) {
				uint8_t gb = (ow[i >> 2] >> (24 - 8 * (i & 3))) & 0xff;
				if (gb != ref[i]) stitchOK = false;
			}
			vk.destroy_buffer(bOff); vk.destroy_buffer(bTot); vk.destroy_buffer(bOut);
		}

		// CPU stitch
		const uint32_t * scr = (const uint32_t *)bScratch.ptr;
		const uint32_t * blen = (const uint32_t *)bBitLen.ptr;
		bw w;
		write_slice_header(w, cfg, 0, 0);
		for (int mb = 0; mb < nmb; ++mb) {
			uint32_t L = blen[mb];
			const uint32_t * base = scr + (size_t)mb * STRIDE;
			for (uint32_t i = 0; i < L; ++i)
				w.put_bit((base[i >> 5] >> (31 - (i & 31))) & 1u);
		}
		w.rbsp_trailing_bits();
		std::vector<uint8_t> frameGPU = build_parameter_sets(cfg);
		emit_nal(frameGPU, 3, NAL_SLICE_IDR, w.bytes());

		bool byteExact = (frameGPU == frameCPU);

		// ffmpeg decode GPU frame, compare to GPU recon
		std::string b = "/tmp/cv264"; FILE * f = fopen((b + ".264").c_str(), "wb");
		fwrite(frameGPU.data(), 1, frameGPU.size(), f); fclose(f);
		int rc = system(("ffmpeg -hide_banner -loglevel error -y -i " + b + ".264 -f rawvideo -pix_fmt yuv420p " + b + ".yuv").c_str());
		int mism = 0; bool dec = (rc == 0);
		if (dec) {
			FILE * yf = fopen((b + ".yuv").c_str(), "rb");
			std::vector<uint8_t> d((size_t)cw * ch * 3 / 2);
			if (fread(d.data(), 1, d.size(), yf) != d.size()) dec = false; fclose(yf);
			if (dec) {
				auto * gY = (int32_t *)bRY.ptr; auto * gCb = (int32_t *)bRCb.ptr; auto * gCr = (int32_t *)bRCr.ptr;
				for (size_t i = 0; i < (size_t)cw * ch; ++i) if (d[i] != (uint8_t)gY[i]) ++mism;
				for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if (d[(size_t)cw * ch + i] != (uint8_t)gCb[i]) ++mism;
				for (size_t i = 0; i < (size_t)cw2 * ch2; ++i) if (d[(size_t)cw * ch + cw2 * ch2 + i] != (uint8_t)gCr[i]) ++mism;
			}
		}
		bool pass = byteExact && dec && mism == 0 && stitchOK;
		if (!pass) ++fails;
		printf("%dx%d qp%d: gpu=%zu cpu=%zu byteExact=%d gpuStitch=%d decode=%s mism=%d  %s\n",
		       t.w, t.h, t.qp, frameGPU.size(), frameCPU.size(), byteExact ? 1 : 0, stitchOK ? 1 : 0, dec ? "ok" : "FAIL", mism, pass ? "PASS" : "FAIL");

		for (auto * bb : {&bSY, &bSC, &bRY, &bRCb, &bRCr, &bLDC, &bLAC, &bCDC, &bCAC, &bNL, &bNC, &bScratch, &bBitLen, &bOrd, &bClaim, &bDone}) vk.destroy_buffer(*bb);
	}
	printf("\n%s (%d failures)\n", fails == 0 ? "GPU CAVLC BYTE-EXACT + DECODE PIXEL-EXACT" : "FAILURES", fails);
	return fails ? 1 : 0;
}
