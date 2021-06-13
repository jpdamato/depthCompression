#pragma once


///////////////////////////////////////////////////
////////////////////////////////////////////
class spline_data
{
public:
	unsigned short x0;
	unsigned short y0;
	unsigned short values_count;
	std::vector<unsigned short> values;
	std::vector<char> cvalues;
	float coefs[4];

	int memSize() { return 6 + values.size(); }

	int last_index;
	int last_value;
	int getValue(int index, int mode)
	{
		if (mode == LOSSLESS_COMPRESSION)
		{
			if (index == last_index + 1)
			{
				last_index++;
				last_value = last_value + cvalues[index];
				return last_value;
			}
			else
			{
				int value = coefs[0];

				for (int i = 0; i <= index; i++) value += cvalues[i];

				last_index = index;
				last_value = value;
				return value;
			}
		}
		else
			if (mode == LINEAR_COMPRESSION || values_count < MIN_SAMPLES_EQ)
			{
				double a = (double)index / values_count;
				int v0 = coefs[0];
				int v1 = coefs[1];
				return (int)(v1 * a + v0 * (1 - a));
			}
			else
			{
				int x = index;
				float v0 = coefs[0] + coefs[1] * x + coefs[2] * x * x + coefs[3] * (x) * (x) * (x);
				return (int)(v0);
			}
	}

	void fit(int mode)
	{
		coefs[0] = coefs[1] = coefs[2] = coefs[3] = 0;


		values_count = values.size();
		if (values.size() == 0) return;

		if (mode == LOSSLESS_COMPRESSION)
		{
			coefs[0] = values[0];
		}
		else
			if (mode == LINEAR_COMPRESSION || values.size() < MIN_SAMPLES_EQ)
			{
				coefs[0] = values[0];
				coefs[1] = values[values_count - 1];
			}
			else
			{
				std::vector<float> xs;
				std::vector<float> ys;

				int subsample = 2;
				for (int i = 0; i < values.size(); i = i + subsample)
				{
					int value = values[i];
					xs.push_back(i);
					ys.push_back(value);
				}

				fitIt((float*)xs.data(), (float*)ys.data(), 3, coefs, xs.size());
			}

	}
};
