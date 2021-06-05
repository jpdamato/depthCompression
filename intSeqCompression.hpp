#pragma once
#include <stdlib.h>
#include<iostream>
#include<fstream>
#include <vector>

#include "timer.h"
#include "vbyte.h"
#include "varintdecode.h"


struct Sorted32Traits 
{
	typedef uint32_t type;
	static constexpr const char *name = "Sorted32";

	static type make_plain_value(size_t i) {
		return i * 7;
	}

	static size_t encode(const type *in, uint8_t *out, size_t length) {
		return vbyte_compress_sorted32(in, out, 0, length);
	}

	static size_t compressed_size(const type *in, size_t length) {
		return vbyte_compressed_size_sorted32(in, length, 0);
	}

	static size_t decode(const uint8_t *in, type *out, size_t length) {
		return vbyte_uncompress_sorted32(in, out, 0, length);
	}

	static type select(const uint8_t *in, size_t length, size_t index) {
		return vbyte_select_sorted32(in, length, 0, index);
	}

	static size_t search(const uint8_t *in, size_t length, type value,
		type *result) {
		return vbyte_search_lower_bound_sorted32(in, length, value, 0, result);
	}

	static size_t append(uint8_t *end, type highest, type value) {
		return vbyte_append_sorted64(end, highest, value);
	}
};

class int32Encoder
{
public:

	
	static void encode(std::vector<short>& input, std::vector<short>& comp, int resample = 1)
	{
	
		int highValues = 0;
		int difValues = 0;
		comp.clear();

		int befValue = 0;
		for (int i = 0; i < input.size(); i++)
		{
			// first of sequence
			int value = input[i];

			if ((befValue > 0) && ( abs(value - befValue) < 128 ) )
			{
				comp.push_back( (value - befValue)/resample );
				befValue = value;
				difValues++;
			}
			else
			{
				comp.push_back(value);
				befValue = value;
				highValues++;
			}

		}

		std::cout << "High values " << highValues << " dif values " << difValues << "\n";
	}

	

	static void decode(std::vector<short>& comp, std::vector<short>& input, int resample = 1)
	{
		int befValue = 0;
		for (int i = 0; i < comp.size(); i++)
		{

			int value = comp[i] ;

			if (value < 128)
			{
				input.push_back(befValue + comp[i] * resample);
				befValue = befValue + comp[i] * resample;
			}
			else
			{
				input.push_back(comp[i]);
				befValue = comp[i];
			}
			
		}


	}

};