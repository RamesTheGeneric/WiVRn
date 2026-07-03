/*
 * WiVRn VR streaming — standalone unit test for the HEVC bit-writer.
 * Build: g++ -std=c++20 -I.. test_bitwriter.cpp ../bitwriter.cpp -o test_bitwriter
 */

#include "../bitwriter.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace wivrn::hevc;

static int failures = 0;

// Render the byte-aligned RBSP of a writer as a bit string, but only the first
// `nbits` bits (so we can compare exact prefixes without padding noise).
static std::string bits_of(bitwriter & w, size_t nbits)
{
	auto & b = w.bytes();
	std::string s;
	for (size_t i = 0; i < nbits; ++i)
	{
		uint8_t byte = b[i / 8];
		s.push_back(((byte >> (7 - (i % 8))) & 1) ? '1' : '0');
	}
	return s;
}

static void expect_eq(const std::string & got, const std::string & want, const char * what)
{
	if (got != want)
	{
		std::printf("FAIL %s: got '%s' want '%s'\n", what, got.c_str(), want.c_str());
		++failures;
	}
	else
	{
		std::printf("ok   %s = %s\n", what, want.c_str());
	}
}

static void check_ue(uint32_t v, const std::string & want)
{
	bitwriter w;
	w.put_ue(v);
	size_t n = w.bit_count();
	expect_eq(bits_of(w, n), want, ("ue(" + std::to_string(v) + ")").c_str());
}

static void check_se(int32_t v, const std::string & want)
{
	bitwriter w;
	w.put_se(v);
	size_t n = w.bit_count();
	expect_eq(bits_of(w, n), want, ("se(" + std::to_string(v) + ")").c_str());
}

int main()
{
	// Exp-Golomb ue(v) reference values (Rec. ITU-T H.265 Table 9-2).
	check_ue(0, "1");
	check_ue(1, "010");
	check_ue(2, "011");
	check_ue(3, "00100");
	check_ue(4, "00101");
	check_ue(5, "00110");
	check_ue(6, "00111");
	check_ue(7, "0001000");
	check_ue(8, "0001001");
	check_ue(14, "0001111");
	check_ue(15, "000010000");

	// Signed mapping se(v).
	check_se(0, "1");
	check_se(1, "010");
	check_se(-1, "011");
	check_se(2, "00100");
	check_se(-2, "00101");
	check_se(3, "00110");
	check_se(-3, "00111");

	// Fixed-length u(n), MSB first.
	{
		bitwriter w;
		w.put_bits(0xA, 4); // 1010
		expect_eq(bits_of(w, 4), "1010", "u(4)=0xA");
	}
	{
		bitwriter w;
		w.put_bits(0x2C, 6); // 101100
		expect_eq(bits_of(w, 6), "101100", "u(6)=0x2C");
	}

	// Emulation prevention: 00 00 00 -> 00 00 03 00, 00 00 01 -> 00 00 03 01.
	{
		std::vector<uint8_t> rbsp = {0x00, 0x00, 0x00, 0x00, 0x01, 0x02};
		std::vector<uint8_t> out;
		append_ebsp(out, rbsp);
		// After each "00 00", a byte <= 0x03 is escaped with an inserted 0x03.
		// Decoding drops the 0x03 after "00 00", recovering the original.
		std::vector<uint8_t> want = {0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x01, 0x02};
		bool eq = out == want;
		std::printf("%s emulation-prevention\n", eq ? "ok  " : "FAIL");
		if (not eq)
		{
			++failures;
			std::printf("  got:");
			for (auto b: out)
				std::printf(" %02X", b);
			std::printf("\n  want:");
			for (auto b: want)
				std::printf(" %02X", b);
			std::printf("\n");
		}
	}

	// NAL emission: start code + header for SPS (nal_unit_type 33 -> 0x42 0x01).
	{
		std::vector<uint8_t> rbsp = {0xAB, 0xCD};
		std::vector<uint8_t> out;
		emit_nal(out, 33, rbsp);
		std::vector<uint8_t> want = {0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0xAB, 0xCD};
		bool eq = out == want;
		std::printf("%s emit_nal(SPS)\n", eq ? "ok  " : "FAIL");
		if (not eq)
			++failures;
	}

	// rbsp_trailing_bits after a byte-aligned payload -> 0x80.
	{
		bitwriter w;
		w.put_bits(0, 0); // nothing
		w.rbsp_trailing_bits();
		auto & b = w.bytes();
		bool eq = b.size() == 1 && b[0] == 0x80;
		std::printf("%s rbsp_trailing_bits(aligned)=0x80\n", eq ? "ok  " : "FAIL");
		if (not eq)
			++failures;
	}

	std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILED", failures);
	return failures == 0 ? 0 : 1;
}
