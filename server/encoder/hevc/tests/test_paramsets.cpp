/*
 * Emit VPS/SPS/PPS for a chosen resolution to stdout (raw Annex-B) so an
 * external parser (ffprobe / dec265) can validate them.
 * Build: g++ -std=c++20 -I.. test_paramsets.cpp ../param_sets.cpp ../bitwriter.cpp -o test_paramsets
 * Use:   ./test_paramsets 1832 1920 8 > ps.265 ; ffprobe ps.265
 */

#include "../param_sets.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

int main(int argc, char ** argv)
{
	wivrn::h267::hevc_config cfg;
	cfg.width = argc > 1 ? std::atoi(argv[1]) : 1832;
	cfg.height = argc > 2 ? std::atoi(argv[2]) : 1920;
	cfg.bit_depth = argc > 3 ? std::atoi(argv[3]) : 8;
	cfg.qp = 26;

	auto ps = wivrn::h267::build_parameter_sets(cfg);

	std::fprintf(stderr,
	             "display %ux%u  coded %ux%u  ctbs %ux%u  bytes %zu\n",
	             cfg.width, cfg.height, cfg.coded_width(), cfg.coded_height(),
	             cfg.ctbs_x(), cfg.ctbs_y(), ps.size());

	ssize_t n = write(STDOUT_FILENO, ps.data(), ps.size());
	return n == (ssize_t)ps.size() ? 0 : 1;
}
