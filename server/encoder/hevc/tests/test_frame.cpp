/*
 * Emit a complete, decodable all-intra IDR HEVC frame using the CPU reference
 * CABAC coder and parameter sets. Every CU is a 2Nx2N intra CU with all coded-
 * block-flags zero, so the picture is flat (DC prediction from default 128).
 * This validates the whole bitstream path (param sets + slice header + CABAC +
 * coding tree + termination) against a reference decoder.
 *
 * Build: g++ -std=c++20 -I.. test_frame.cpp ../param_sets.cpp ../bitwriter.cpp \
 *            ../cabac_reference.cpp -o test_frame
 * Use:   ./test_frame 256 256 > f.265 ; dec265 f.265 ; ffmpeg -i f.265 out.png
 */

#include "../cabac_reference.h"
#include "../param_sets.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

using namespace wivrn::hevc;
using namespace wivrn::hevc::tables;

// Encode one 64x64 CTB coded as a single 2Nx2N intra CU with no residual.
static void encode_flat_ctb(cabac_encoder & cb)
{
	// coding_quadtree(log2CbSize = 6): do not split. ctxInc = 0 (no neighbour is
	// deeper than the current tree depth, since every CTB is a single CU).
	cb.encode_bin(SPLIT_CODING_UNIT_FLAG + 0, 0);

	// coding_unit: I-slice intra, log2CbSize(6) != MinCbLog2Size so part_mode is
	// inferred 2Nx2N; pcm disabled.
	// Luma intra mode via MPM: prev_intra_luma_pred_flag = 1, mpm_idx = 1. With
	// all reference samples equal to 128 the reconstructed mode is irrelevant —
	// every intra mode predicts a flat 128 block.
	cb.encode_bin(PREV_INTRA_LUMA_PRED_FLAG, 1);
	cb.encode_bypass(1); // mpm_idx truncated-unary "10" -> value 1
	cb.encode_bypass(0);

	// intra_chroma_pred_mode = 4 (derived mode): single context-coded 0 bin.
	cb.encode_bin(INTRA_CHROMA_PRED_MODE + 0, 0);

	// transform_tree(log2TrafoSize = 6, trafoDepth = 0): split is implicit
	// (6 > MaxTbLog2SizeY = 5), so split_transform_flag is not coded. Chroma
	// cbf_cb / cbf_cr are coded at depth 0 (both zero -> no chroma residual and
	// children inherit zero).
	cb.encode_bin(CBF_CB_CR + 0, 0); // cbf_cb
	cb.encode_bin(CBF_CB_CR + 0, 0); // cbf_cr

	// Four 32x32 child TUs (trafoDepth = 1), each a leaf with cbf_luma = 0.
	for (int i = 0; i < 4; ++i)
		cb.encode_bin(CBF_LUMA + 0, 0); // ctxInc = (trafoDepth==0)?1:0 = 0
}

int main(int argc, char ** argv)
{
	hevc_config cfg;
	cfg.width = argc > 1 ? std::atoi(argv[1]) : 256;
	cfg.height = argc > 2 ? std::atoi(argv[2]) : 256;
	cfg.bit_depth = 8;
	cfg.qp = 26;

	std::vector<uint8_t> stream = build_parameter_sets(cfg);

	// Slice: single slice covering the whole picture.
	bitwriter hdr;
	write_slice_header(hdr, cfg, /*slice_ctb_addr*/ 0, /*first_in_pic*/ true);
	auto rbsp = hdr.bytes(); // byte aligned after byte_alignment()

	cabac_encoder cb;
	cb.init(/*init_type I*/ 0, cfg.qp);

	const uint32_t n_ctbs = cfg.ctbs_total();
	for (uint32_t i = 0; i < n_ctbs; ++i)
	{
		encode_flat_ctb(cb);
		// end_of_slice_segment_flag: 1 on the last CTB (flushes the engine).
		cb.encode_terminate(i + 1 == n_ctbs ? 1 : 0);
	}
	auto cabac_bytes = cb.finish();
	rbsp.insert(rbsp.end(), cabac_bytes.begin(), cabac_bytes.end());

	// Wrap the slice RBSP in an IDR NAL and append to the stream.
	emit_nal(stream, NAL_IDR_W_RADL, rbsp);

	std::fprintf(stderr,
	             "display %ux%u coded %ux%u ctbs %ux%u=%u  slice_rbsp %zu bytes  total %zu bytes\n",
	             cfg.width, cfg.height, cfg.coded_width(), cfg.coded_height(),
	             cfg.ctbs_x(), cfg.ctbs_y(), n_ctbs, rbsp.size(), stream.size());

	ssize_t w = write(STDOUT_FILENO, stream.data(), stream.size());
	return w == (ssize_t)stream.size() ? 0 : 1;
}
