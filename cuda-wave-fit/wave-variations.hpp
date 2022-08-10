#pragma once

#include <vector>

struct WaveVariations
{

    size_t runs = 0;

	std::vector<size_t> factors;
	std::vector<size_t> offsets;
	std::vector<double> variations;

    WaveVariations(std::vector<std::vector<double>> permutate)
	{

        size_t offset = 0;
        size_t factor = 1;

        // Add variations and offsets from left to right
        for (size_t i = 0; i < permutate.size(); i += 1)
        {
            // Flatten variations
            variations.insert(
                variations.end(),
                permutate[i].begin(),
                permutate[i].end());
            // Prepend current factor
            offsets.insert(
                offsets.end(),
                offset);
            // Multiply factor by size
            offset += permutate[i].size();
        }

        // Calculate factors from right to left
        for (size_t i = permutate.size() - 1; i != -1; i -= 1)
        {
            // Prepend current factor
            factors.insert(
                factors.begin(),
                factor);
            // Multiply factor by size
            factor *= permutate[i].size();
        }

        runs = permutate.empty() ? 0 : factor;
    }

};