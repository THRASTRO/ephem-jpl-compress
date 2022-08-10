#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdarg.h>

#include "inc/cuda-wave-fit.h"

/* Print an error message. */
void err(const char* msg, ...)
{
    fputs("Error: ", stderr);
    va_list ap;
    va_start(ap, msg);
    vfprintf(stderr, msg, ap);
    va_end(ap);
    putc('\n', stderr);
}

size_t LoadData(const char* path, double*& data, bool err)
{

    size_t points = std::filesystem::file_size(path) / 8;
    std::fstream dfs(path, std::fstream::in | std::fstream::binary);

    // Check to see that the file was opened correctly:
    if (!dfs || !dfs.is_open()) {
        if (err)
        {
            std::cerr << "There was a problem opening the input file!\n";
            exit(1); // Exit or do additional error checking
        }
        else
        {
            // Indicate error only
            return std::string::npos;
        }
    }

    char* dbytes = new char[8 * points];
    dfs.read(dbytes, points * 8);

    if (sizeof(dtype) != sizeof(double))
    {
        printf("Translate to float32\n");
        // assume dtype is now float
        double* data64 = (double*)dbytes;
        char* dbytes32 = new char[sizeof(float) * points];
        data = (dtype*)dbytes32;
        std::transform(data64, data64 + points, data,
            [](double d) -> dtype {return dtype(d); });
        delete[] dbytes;
        dbytes = dbytes32;
    }
    else
    {
        data = (dtype*)dbytes;
    }
    
    return points;
}

void SaveFrames(const std::vector<FrameTerms>& frames, const std::string& path)
{
    std::fstream fs(path, std::fstream::out | std::fstream::binary);
    dtype size = dtype(frames.size());
    fs.write(reinterpret_cast<char*>(&size), sizeof(dtype));
    for (size_t f = 0; f < size; f += 1)
    {
        dtype terms = dtype((frames[f].size() - 2) / 3);
        fs.write(reinterpret_cast<const char*>(&terms), sizeof(dtype));
        dtype min = frames[f][frames[f].size() - 2];
        dtype max = frames[f][frames[f].size() - 1];
        fs.write(reinterpret_cast<const char*>(&min), sizeof(dtype));
        fs.write(reinterpret_cast<const char*>(&max), sizeof(dtype));
        fs.write(reinterpret_cast<const char*>(frames[f].data()), sizeof(dtype) * (frames[f].size() - 2));
    }
    fs.flush();
    fs.close();
}

void SaveWaves(const std::vector<dtype>& terms, const std::string& path)
{
    std::fstream fs(path, std::fstream::out | std::fstream::binary);
    dtype size = 1; // dtype(waves.size());
    fs.write(reinterpret_cast<char*>(&size), sizeof(dtype));
    size = dtype(terms.size() - 2) / 3;
    fs.write(reinterpret_cast<const char*>(&size), sizeof(dtype));
    dtype min = terms[terms.size() - 2];
    dtype max = terms[terms.size() - 1];
    fs.write(reinterpret_cast<const char*>(&min), sizeof(dtype));
    fs.write(reinterpret_cast<const char*>(&max), sizeof(dtype));
    fs.write(reinterpret_cast<const char*>(terms.data()), sizeof(dtype) * (terms.size() - 2));
    fs.flush();
    fs.close();
}
