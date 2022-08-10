#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>

#include "inc/cuda-wave-fit.h"
#include "opencv2/core/core.hpp"
#include "wave-variations.hpp"
#include "cuda-array.hpp"

#include "getopts.hpp"
#include <numeric>
#include <unordered_set>

#define NOMINMAX
#include <Windows.h>
#include <tchar.h>

int device = 0;

bool optimize_on_start = true;
bool optimize_all = true;
bool use_angles = false;
// bool optimize_on_start = false;

double amp_cutoff = 0.01;
double abort_err = 1e-3;

int min_opt_runs = 6;

int guess_terms_e_from = -32;
int guess_terms_e_to = 0;
int guess_terms_e_step = 1;

int guess_params_e_from = -65;
int guess_params_e_to = 0;
int guess_params_e_step = 1;

size_t max_guess_trials = 512;

size_t opt_load_max_runs = 50000;
double opt_load_limiter = 0.995;
double opt_load_limit = 0.00001;

size_t opt_frames_max_runs = 25000;
double opt_frames_limiter = 0.975;
double opt_frames_limit = 0.0001;

size_t opt_frame_max_runs = 750;
double opt_frame_limiter = 0.95;
double opt_frame_limit = 1e-9;

int max_guesses = 16;
double freq_step = 0.0125;
double phase_step = 0.02;

// double amp_steps = 24;
// double amp_factor = 0.923;

size_t max_frame_terms = 32;
size_t max_waves_per_frame = 32;


// Additional command to print help page to screen
void getopt_help_cmd(struct CmdGetOpt* getopt, union CmdOptionValue value)
{
    std::stringstream strm;
    getopt_print_help(getopt, strm);
    fprintf(stderr, "Usage: jpl-extract [options] DATABASE BODY\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, strm.str().c_str());
    exit(0);
}

std::vector<dtype> LoadArray(const std::string& input)
{
    std::ifstream fs(input, std::fstream::binary);
    std::vector<dtype> numbers;
    std::vector<char> buffer(1024, 0);
    while (fs && !fs.eof()) {
        fs.read(buffer.data(), buffer.size());
        std::streamsize readed = fs.gcount();
        double* dview = reinterpret_cast<double*>(buffer.data());
        // ToDo: check if readed bytes is dividable by 8
        // Otherwise we know we have a faulty save here!
        numbers.insert(numbers.end(), dview,
            dview + readed / sizeof(dtype));
    }
    return numbers;
}

std::vector<FrameTerms> LoadFrames(const std::string& input)
{
    std::vector<dtype> arr = LoadArray(input);
    if (arr.empty()) arr.push_back(0);
    size_t n_frames = size_t(arr.front());
    std::vector<FrameTerms> frames(n_frames);
    for (size_t f = 0, i = 1; f < n_frames; f += 1)
    {
        size_t n_terms = size_t(arr[i++]);
        dtype min = arr[i++], max = arr[i++];
        for (size_t t = 0; t < n_terms; t += 1)
        {
            frames[f].push_back(arr[i++]);
            frames[f].push_back(arr[i++]);
            frames[f].push_back(arr[i++]);
        }
        frames[f].push_back(min);
        frames[f].push_back(max);
    }
    return frames;
}


std::vector<dtype> LoadFrame(const std::string& input)
{
    std::vector<dtype> arr = LoadArray(input);
    arr.erase(arr.begin()); // Remove frame count
    arr.erase(arr.begin()); // Remove term count
    dtype min = arr.front(); arr.erase(arr.begin());
    dtype max = arr.front(); arr.erase(arr.begin());
    arr.push_back(min);
    arr.push_back(max);
    return arr;
}

template <typename T>
void interp(T start, T end, double smin, double smax, double dmin, double dmax)
{
    double srange = (smax - smin);
    double drange = (dmax - dmin);
    double factor = drange / srange;
    for (start; start != end; start++)
        *start = (*start - smin) * factor + dmin;
}
/*
double GetFramesScore(const CudaArray<dtype>& data, const std::vector<dtype>& flat)
{
    double rv = CudaFramesScore(data, flat);
    // printf("Got delta frames from GPU %g\n", rv / data.size);
    return rv / data.size;
}
*/

double GetFramesScore(const DataPoints& data, const std::vector<dtype>& flat)
{
    // std::vector<dtype> data(org.begin(), org.begin() + 2);
    // double rv = CudaFramesScore(data, flat);
    // one frame, zero terms, min and max
    if (flat.size() < 4) return INVSCORE;
    double delta = 0.0;
    double dt = 1.0 / (data.size() - 1);
    for (int p = 0; p < data.size(); p += 1)
    {
        double ts = double(p) * dt;
        size_t i = 0; double value = 0.0;
        size_t frames = size_t(flat[i++]);
        for (size_t f = 0; f < frames; f += 1)
        {
            double dada = 0.0;
            size_t terms = size_t(flat[i++]);
            for (size_t t = 0; t < terms; t += 1, i += 3)
            {
                // Optimizer Bug with [i++]!!!!
                // MSVC changes to arr[i++] + ts * flat[i++] !!!!!
                // Which leads to completely wrong results, WTF !!!!!
                dada += sin((ts * flat[i + 0]
                    + flat[i + 1])) * flat[i + 2];
            }
            double min = flat[i++], max = flat[i++];
            // value += min + (dada + 1) / 2 * (max - min);
            value = min + (value + dada + 1) / 2 * (max - min);
        }
        // printf("CPU Add %g\n", abs(data[p] - value));
        delta += abs(data[p] - value);
    }
    // printf("GPU off was %.22g\n", rv - delta);
    //if (abs(rv - delta) > 1e-3)
    //{
    //    printf("Score1 does not agree %.22g\n",
    //        rv - delta);
    //}

    // Normalize score into -1 to +1 range
    // double min = flat[flat.size() - 2];
    // double max = flat[flat.size() - 1];
    return delta / data.size();
}

/*

double GetFrameScore(const DataPoints& data, const CudaArray<dtype>& _data, const FrameTerms& terms)
{
    return CudaFrameScore(_data, terms) / _data.size;

    // printf("Got delta frames from GPU %g\n", rv[0]);
    // double gpu = GetGPUFrameScore(data, terms);
    double delta = 0;
    double dt = 1.0 / (data.size() - 1);
    for (int p = 0; p < data.size(); p += 1)
    {
        double value = 0.0;
        double ts = double(p) * dt;
        for (size_t t = 0; terms.size() > 1 && t < terms.size() - 2;)
        {
            value += sin((ts * terms[t++]
                + terms[t++])) * terms[t++];
        }
        delta += abs(data[p] - value);
    }

    if (true)
    {
        double rv = CudaFrameScore(_data, terms);
        if (abs(rv - delta) > 1e-3)
        {
            printf("Score2 does not agree %.22g\n",
                rv - delta);
        }
        return rv / data.size();
    }
    // printf("GPU off was %.22g\n", rv - score);
    // exit(1);
    return delta / data.size();

    // return rv / _data.size;
}

double GetFrameScore(const CudaArray<dtype>& data, const FrameTerms& terms)
{
    double rv = CudaFrameScore(data, terms);
    printf("Got delta frames from GPU %g\n", rv);
    return rv / data.size;
}
*/

double GetFrameScore(const DataPoints& data, const FrameTerms& terms)
{
    // double gpu = GetGPUFrameScore(data, terms);
    double score = 0;
    double dt = 1.0 / (data.size() - 1);
    for (int p = 0; p < data.size(); p += 1)
    {
        double value = 0.0;
        double ts = double(p) * dt;
        for (size_t t = 0; terms.size() > 1 && t < terms.size() - 2; t += 3)
        {
            value += sin((ts * terms[t+0]
                + terms[t+1])) * terms[t+2];
        }
        score += abs(data[p] - value);
    }
    // printf("GPU was %g vs mine %g\n", gpu, score);
    // exit(1);
    return score / data.size();
}

dtype GetAngle(dtype prv, dtype cur, dtype nxt)
{
    dtype ab = abs(prv) - abs(cur), bc = abs(cur) - abs(nxt);
    dtype angle = abs(atan2(ab - bc, 1 + ab * bc));
    return angle > PI ? PI - angle : angle;
}

void AddGuess(std::unordered_map<dtype, dtype>& map, double value, double amplitude)
{
    if (map.contains(value))
    {
        map[value] = std::max(map[value], amplitude);
    }
    else 
    {
        map.insert(std::make_pair(value, amplitude));
    }
}

std::unordered_map<dtype, dtype> GuessFrequenciesFFT(const DataPoints& data, double max_amp32, size_t max = 50, double cuting = 0.0)
{
    size_t points = data.size();

    // Create the fourier transformation
    typedef std::complex<dtype> complex;
    std::vector<complex> fft;
    cv::dft(std::vector<complex>(data.begin(), data.end()),
        fft, cv::DftFlags::DFT_COMPLEX_OUTPUT);

    DataPoints amps(points); // amplitudes
    DataPoints angles(points); // abs angles
    DataPoints rangles(points); // real angles
    DataPoints iangles(points); // imag angles


    std::fill(amps.begin(), amps.end(), INVSCORE);
    std::fill(angles.begin(), angles.end(), INVSCORE);
    std::fill(rangles.begin(), rangles.end(), INVSCORE);
    std::fill(iangles.begin(), iangles.end(), INVSCORE);

    Indicies samps(points);
    Indicies sangles(points);
    Indicies srangles(points);
    Indicies siangles(points);

    // Create indicies counting up from zero
    std::iota(samps.begin(), samps.end(), 0);
    std::iota(sangles.begin(), sangles.end(), 0);
    std::iota(srangles.begin(), srangles.end(), 0);
    std::iota(siangles.begin(), siangles.end(), 0);

    DataPoints freqs(points);
    dtype freq = dtype(0.5 * points - 0.5);
    std::generate(freqs.begin(), freqs.end(),
        [&freq] { return abs(freq--); });

    for (size_t i = 0; i < points; i++)
    {
        //rangles[i] = GetAngle(fft[i - 1].real(), fft[i].real(), fft[i + 1].real());
        //iangles[i] = GetAngle(fft[i - 1].imag(), fft[i].imag(), fft[i + 1].imag());
        //angles[i] = GetAngle(abs(fft[i - 1]), abs(fft[i]), abs(fft[i + 1]));
        amps[i] = abs(fft[i]);

    }

    /*
    samps.erase(samps.begin(), samps.begin() + std::min(cutoff, samps.size()));
    srangles.erase(srangles.begin(), srangles.begin() + std::min(cutoff, srangles.size()));
    siangles.erase(siangles.begin(), siangles.begin() + std::min(cutoff, siangles.size()));
    sangles.erase(sangles.begin(), sangles.begin() + std::min(cutoff, sangles.size()));

    samps.erase(samps.begin() + std::min(samps.size() - cutoff, size_t(0)), samps.end());
    srangles.erase(srangles.begin() + std::min(srangles.size() - cutoff, size_t(0)), srangles.end());
    siangles.erase(siangles.begin() + std::min(siangles.size() - cutoff, size_t(0)), siangles.end());
    sangles.erase(sangles.begin() + std::min(sangles.size() - cutoff, size_t(0)), sangles.end());
    */

    samps.erase(samps.begin() + std::max(samps.size() / 2, size_t(0)), samps.end());
    srangles.erase(srangles.begin() + std::max(srangles.size() / 2, size_t(0)), srangles.end());
    siangles.erase(siangles.begin() + std::max(siangles.size() / 2, size_t(0)), siangles.end());
    sangles.erase(sangles.begin() + std::max(sangles.size() / 2, size_t(0)), sangles.end());

    std::sort(samps.begin(), samps.end(),
        [&amps](const auto lhs, const auto rhs)
        { return amps[lhs] > amps[rhs]; });
    std::sort(srangles.begin(), srangles.end(),
        [&rangles](const auto lhs, const auto rhs)
        { return rangles[lhs] > rangles[rhs]; });
    std::sort(siangles.begin(), siangles.end(),
        [&iangles](const auto lhs, const auto rhs)
        { return iangles[lhs] > iangles[rhs]; });
    std::sort(sangles.begin(), sangles.end(),
        [&angles, &amps](const auto lhs, const auto rhs)
        { return (angles[lhs] * amps[lhs]) > (angles[rhs] * amps[rhs]); });

    dtype N = dtype(points);
    dtype dt = dtype(1.0 / N);
    dtype T = dtype(dt * N);
    dtype df2 = dtype(1.0 / T);

    // Make sure guesses are unique
    std::unordered_map<dtype, dtype> guesses;

    // Always include correction term
    // ToDo: Not sure if really needed
    // guesses.insert(0.5);

    printf("Dominating frequency found at %d with amplitude %.3g\n",
        int(samps[0]), amps[samps[0]] / points * 2);

    dtype dmin = *std::min_element(data.begin(), data.end());
    dtype dmax = *std::max_element(data.begin(), data.end());
    double dmax_amp = std::max(abs(dmin), abs(dmax));

    double max_amp = 2.0 / dmax_amp / points;

    double max_amp_detected = amps[samps[0]];

    max_amp = (1.0 / amps[samps[0]]) * max_amp32;
    
        // exit(1);
    // double max_freq = int(samps[0]) * 1.75;
    size_t max_freq = data.size() / 2.0;

    // Now insert the best choices we found
    for (int r = 0; r < points / 2; r += 1)
    {
        if (max_amp_detected * amp_cutoff < amps[samps[r]] && samps[r] < max_freq)
        {
            // ToDo: trim possible frequencies?
            double amplitude = (amps[samps[r]] * max_amp);
            AddGuess(guesses, dtype(samps[r]) - 0.5, amplitude);
            AddGuess(guesses, dtype(samps[r]) + 0.5, amplitude);
        }
        if (guesses.size() >= max) break;
        if (use_angles == false) continue;

        if (max_amp_detected * amp_cutoff < amps[sangles[r]] && sangles[r] < max_freq)
        {
            double amplitude = (amps[sangles[r]] * max_amp);
            AddGuess(guesses, dtype(sangles[r]) - 0.5, amplitude);
            AddGuess(guesses, dtype(sangles[r]) + 0.5, amplitude);
            if (guesses.size() >= max) break;
        }

        if (max_amp_detected * amp_cutoff < amps[srangles[r]] && srangles[r] < max_freq)
        {
            double amplitude = (amps[srangles[r]] * max_amp);
            AddGuess(guesses, dtype(srangles[r]) - 0.5, amplitude);
            AddGuess(guesses, dtype(srangles[r]) + 0.5, amplitude);
            if (guesses.size() >= max) break;
        }

        if (max_amp_detected * amp_cutoff < amps[siangles[r]] && siangles[r] < max_freq)
        {
            double amplitude = (amps[siangles[r]] * max_amp);
            AddGuess(guesses, dtype(siangles[r]) - 0.5, amplitude);
            AddGuess(guesses, dtype(siangles[r]) + 0.5, amplitude);
            if (guesses.size() >= max) break;
        }
    }

    // Convert to vector once all is set
    return guesses;
}

template <typename T>
std::vector<T> RangeIncl(T min, T max, T step)
{
    std::vector<T> range;
    while (min <= max)
    {
        range.push_back(min);
        min += step;
    }
    return range;
}

template <typename T>
std::vector<T> RangeExcl(T min, T max, T step)
{
    std::vector<T> range;
    if (step < 0)
    {
        while (max < min + step / 2.0)
        {
            range.push_back(min);
            min += step;
        }
    }
    else
    {
        while (min < max - step / 2.0)
        {
            range.push_back(min);
            min += abs(step);
        }
    }
    return range;
}

template <typename T>
std::vector<T> HalfSteps(T from, T steps, T factor = 0.5)
{
    std::vector<T> range;
    while (steps-- > 0)
    {
        range.push_back(from);
        from *= factor;
    }
    return range;
}

struct WaveResult
{
    dtype freq;
    dtype phase;
    dtype scale;
    dtype score = INVSCORE;
};

std::vector<dtype> GuessNextFrameFrequency(
    const DataPoints& data, double max_amp, double cuting)
{
    // Get starting score before fitting
    // We basically start with a flat line
    // double start_score = score;
    // Guess frequencies by doing FFT analysis
    // Fetch minima and maxima to determine maximum amplitude

    auto gmap = GuessFrequenciesFFT(data, max_amp, max_guesses, cuting);
    std::vector<std::pair<dtype, dtype>> guesses(gmap.begin(), gmap.end());
    std::sort(guesses.begin(), guesses.end(),
        [](const auto lhs, const auto rhs)
        { return abs(lhs.second - rhs.second) < 1e-4 ? lhs.first < rhs.first : lhs.second > rhs.second; });

    // std::sort(guesses.rbegin(), guesses.rend());

    // Report waves we will probe
    std::stringstream ss; double comma = false;
    for (std::pair<dtype, dtype> guess : guesses)
    {
        if (comma) ss << ", ";
        ss << std::setprecision(24)
           << guess.first << " ("
           << std::setprecision(4)
           << guess.second << ")";
        comma = true;
    }


    // Collect frequencies to try
    std::vector<dtype> frequencies;
    std::vector<dtype> amp_scales;
    for (auto guess : guesses)
    {
        for (dtype freq = guess.first - 0.5; freq <= guess.first + 0.5; freq += freq_step)
        {
            frequencies.push_back(freq);
            amp_scales.push_back(guess.second);
        }
    }



    // frequencies.push_back(100096.8);
    // amp_scales.push_back(0.9625);

    // Add minor correctional terms
    for (int i = -48; i <= +2; i += 2)
    {
        frequencies.push_back(pow(2, i));
        amp_scales.push_back(max_amp); // uh, how to assume?
    }

    // Always check the full range in equal steps
    std::vector<dtype> phases = RangeExcl<double>(0, 1, phase_step);

    // Scale can be approximated a little and should diverge on half steps
    //std::vector<dtype> scales = HalfSteps<double>(1, amp_steps, amp_factor);
    std::vector<dtype> scales = RangeExcl<double>(1, 0, -0.0125);

    for (size_t i = 0; i < phases.size(); i += 1) phases[i] *= TAU;
    for (size_t i = 0; i < frequencies.size(); i += 1) frequencies[i] *= TAU;

    // Create variations to pass to GPU for evaluation
    WaveVariations trials({ frequencies, phases, scales });

    printf("#########################################################################\n");
    printf("Probing waves %s\n", ss.str().c_str()); // Never sure if this is safe!?
    printf("#########################################################################\n");
    printf("Checking %zd variations to beat avg error [%.4g]\n", trials.runs,
        GetFrameScore(data, { 0 }));
    printf("#########################################################################\n");

    //exit(1);

    // return { guesses[0].first * TAU, 0.5 * TAU, guesses[0].second };

    // This will automatically upload all data
    auto rv = CudeFrameTrials(
        data, trials.factors, trials.offsets,
        trials.variations, amp_scales, trials.runs);
    //exit(1);
    return rv;
}


double GuessParameter(const DataPoints& data, FrameTerms& frame,
    size_t param, int from, int to, int step)
{

    // printf("Guess %g\n", wave.freq);
    // Update the initial score (just in case)
    double score = GetFrameScore(data, frame);

    // WaveOptions first = wave;
    // redo:
    // Start with smallest possible offset

    //2.41e-15 // -22
    //1.84e-15 // -16
    double direction = 0.0;
    FrameTerms up, down;
    up = frame, down = frame;
    for (int e = from; e < to; e += step)
    {
        // Find direction in which to go
        double offset = pow(2, e);
        up[param] = frame[param] + offset;
        down[param] = frame[param] - offset;
        double up_score = GetFrameScore(data, up);
        double down_score = GetFrameScore(data, down);
        if (score > up_score && score > down_score) {
            // printf("Both directions are better?\n");
            if (up_score < down_score) {
                direction = +offset;
                break;
            }
            direction = -offset;
            break;
        }
        if (score > up_score) {
            direction = +offset;
            break;
        }
        if (score > down_score) {
            direction = -offset;
            break;
        }
    }

    // printf("%g\n", direction);

    // Found no direction to go to
    // Means initial wave was best
    // if (direction == 0.0)exit(1);
    if (direction == 0.0) return score;

    // Found direction and minimal magnitude
    // Now search for upper limit of improve
    double multiplyer = 1;
    dtype start_param = frame[param];
    // Optimized by 3.2278% (28 runs in 172.16s)
    for (size_t trials = 0; trials < max_guess_trials; trials += 1)
    {
        double old_param = frame[param];
        frame[param] = start_param + direction * multiplyer;
        double cur_score = GetFrameScore(data, frame);
        // Score did improve
        if (score > cur_score)
        {
            multiplyer *= 2.0;
            score = cur_score;
            continue;
        }
        // Score regressed
        else if (cur_score > score)
        {
            frame[param] = old_param;
            return score;
            // goto redo;
        }
        // Score stayed the same
        else
        {
            // Update parameter to point to new position
            // Try to add the same amount again and re-check
            start_param += direction * multiplyer;
            continue;
        }
    }
    // Not sure we ever meet this?
    printf("Exhausted trials\n");
    // Return what we have
    return score;
}


double GuessFrameParameters(const DataPoints& data, std::vector<dtype>& flat,
    size_t param, int from = -62, int to = -2, int step = 1)
{

    // printf("Guess %g\n", wave.freq);
    // Update the initial score (just in case)
    double score = GetFramesScore(data, flat);

    // WaveOptions first = wave;
// redo:
    // Start with smallest possible offset

    //2.41e-15 // -22
    //1.84e-15 // -16
    double direction = 0.0;
    std::vector<dtype> up, down;
    up = flat, down = flat;
    for (int e = from; e < to; e += step)
    {
        // Find direction in which to go
        double offset = pow(2, e);
        up[param] = flat[param] + offset;
        down[param] = flat[param] - offset;
        double up_score = GetFramesScore(data, up);
        double down_score = GetFramesScore(data, down);
        if (score > up_score && score > down_score) {
            // printf("Both directions are better?\n");
            if (up_score < down_score) {
                direction = +offset;
                break;
            }
            direction = -offset;
            break;
        }
        if (score > up_score) {
            direction = +offset;
            break;
        }
        if (score > down_score) {
            direction = -offset;
            break;
        }
    }

    // printf("%g\n", direction);

    // Found no direction to go to
    // Means initial wave was best
    // if (direction == 0.0)exit(1);
    if (direction == 0.0) return score;

    // Found direction and minimal magnitude
    // Now search for upper limit of improve
    double multiplyer = 1;
    dtype start_param = flat[param];
    // Optimized by 3.2278% (28 runs in 172.16s)
    for (size_t trials = 0; trials < 128; trials += 1)
    {
        double old_param = flat[param];
        flat[param] = start_param + direction * multiplyer;
        double cur_score = GetFramesScore(data, flat);
        // Score did improve
        if (score > cur_score)
        {
            multiplyer *= 2.0;
            score = cur_score;
            continue;
        }
        // Score regressed
        else if (cur_score > score)
        {
            flat[param] = old_param;
            return score;
            // goto redo;
        }
        // Score stayed the same
        else
        {
            // Update parameter to point to new position
            // Try to add the same amount again and re-check
            start_param += direction * multiplyer;
            continue;
        }
    }
    // Not sure we ever meet this?
    printf("Exhausted trials\n");
    // Return what we have
    return score;
}


FrameTerms OptimizeFrameTerms(
    const DataPoints& data,
    const FrameTerms& terms,
    size_t opt_max_runs,
    dtype opt_limiter,
    dtype opt_limit)
{

    double start_score = GetFrameScore(data, terms);

    printf("-------------------------------------------------------------------------\n");
    printf("Optimizing %zd wave(s) with avg error [%.4g]\n",
        (size_t(terms.size()) - 2) / 3, start_score);
    for (size_t i = 0; i < terms.size() - 2; i += 3) {
        std::vector test = { terms[i], terms[i + 1], terms[i + 2] };
        double self_score = GetFrameScore(data, test);
        printf(" %02zd) [%.4g] <%.4g%%> sin(%.12g + %.12g) * %.12g\n",
            (size_t(terms.size()) - 2 - i) / 3, start_score, start_score / self_score * 100.0 * sqrt((terms.size() - 2) / 3),
            terms[i] / TAU, terms[i + 1] / TAU, terms[i + 2]);
    }

    size_t runs = 0; // count
    clock_t start_ts = clock();
    FrameTerms winner = terms;

    // Start limiter high to allow some loops
    double limiter = pow(opt_frame_limiter, - min_opt_runs);

    size_t max_non_improve = winner.size() * 2 + 1;
    size_t has_non_improve = 0;

    double last_save = start_ts;
    double save_interval = CLOCKS_PER_SEC * 10;
    double last_reported = start_score;

    size_t p = 0;

   //opt_max_runs *= winner.size();

    double score = start_score; // Remember

                                // Do a maximum amount of loops (bail out when needed)
    for (runs = 1; runs <= opt_max_runs; runs += 1, p += 1)
    {

        // Roll over the parameter
        // ToDo: maybe do it randomly
        if (p == winner.size() - 2) p = 0;
        // FrameTerms current = winner;
        // Make all parameters available for guessing
        // for (size_t p = 0; p < winner.size() - 2; p += 1)
        double best = GuessParameter(data, winner,
                p, guess_terms_e_from, guess_terms_e_to, guess_terms_e_step);

        // ToDo: optimize to not always recalculate
        //double best = GetFrameScore(data, winner);
        // Bail if not improved
        if (best >= score) {
            has_non_improve++;
            if (has_non_improve > max_non_improve) {
                // printf("Too many non improves\n");
                break;
            }
        }
        else
        {
            has_non_improve = 0;

            clock_t now_ts = clock();
            if (now_ts > last_save + save_interval)
            {
                printf("Optimized by %.4g%% (-%.4g) [%.4g] (%zd runs in %.2fs)\n",
                    (start_score - best) / start_score * 100.0, last_reported - best, best,
                    runs, (double(clock()) - start_ts) / CLOCKS_PER_SEC);
                last_reported = best;
                last_save = now_ts;
            }

            // Sum up all optimizations 
            limiter = (limiter + (score - best) / score) * opt_limiter;
            //printf("  optimized by %g\n", score - best);
            // Update best score
            // winner = current;
            score = best;
            // Bail out when limit is reached
            if (limiter < opt_limit) {
                // printf("Limiter reached\n");
                break;
            }

        }
    }

    printf("-------------------------------------------------------------------------\n");
    printf("Optimized by %.4g%% (-%.4g) [%.4g] (%zd runs in %.2fs)\n",
        (start_score - score) / start_score * 100.0, start_score - score, score,
        runs, (double(clock()) - start_ts) / CLOCKS_PER_SEC);
    for (size_t i = 0; i < winner.size() - 2; i += 3) {
        std::vector test = { winner[i], winner[i + 1], winner[i + 2] };
        double self_score = GetFrameScore(data, test);
        printf(" %02zd) [%.4g] <%.4g%%> sin(%.12g + %.12g) * %.12g\n",
            (size_t(winner.size()) - 2 - i) / 3, score, score / self_score * 100.0 * sqrt((terms.size() - 2) / 3),
            winner[i] / TAU, winner[i + 1] / TAU, winner[i + 2]);
    }

    return winner;
}


std::vector<FrameTerms> FlatToFrames(const std::vector<dtype>& flat)
{
    std::vector<FrameTerms> frames(size_t(flat.front()));
    for (size_t f = 0, i = 1; f < frames.size(); f += 1)
    {
        size_t terms = size_t(flat[i++]);
        size_t params = terms * 3 + 2;
        frames[f].insert(
            frames[f].end(),
            flat.begin() + i,
            flat.begin() + i + params);
        i += params;
    }
    return frames;
}

std::vector<dtype> FramesToFlat(const std::vector<FrameTerms>& frames)
{
    std::vector<dtype> flat;
    flat.push_back(dtype(frames.size()));
    for (const FrameTerms& frame : frames)
    {
        flat.push_back(dtype((frame.size() - 2) / 3));
        flat.insert(flat.end(), frame.begin(), frame.end());
    }
    return flat;
}

void SaveWaves(const std::vector<dtype>& terms, const std::string& path);
void SaveFrames(const std::vector<FrameTerms>& frames, const std::string& path);

FrameTerms OptimizeFrames(
    const DataPoints& data,
    const std::vector<dtype> flat,
    double max_amp,
    const std::string& path,
    size_t opt_max_runs,
    dtype opt_limiter,
    dtype opt_limit)
{

    if (flat.size() < 5) return flat;
    double score = GetFramesScore(data, flat);
    double start_score = score; // Remember

    // All params to play with
    std::vector<size_t> ps;

    // Make all parameters available for guessing
    // Skip frame sizes and number of frame terms
    for (size_t f = 0, i = 1; f < flat[0]; f += 1)
    {
        size_t terms = size_t(flat[i++]);
        for (size_t t = 0; t < terms; t += 1)
        {
            ps.push_back(i++);
            ps.push_back(i++);
            ps.push_back(i++);
        }
        ps.push_back(i++);
        ps.push_back(i++);
    }

    printf("=========================================================================\n");
    printf("Optimizing %zd parameters with delta score {%.8g}\n",
        ps.size() /*flat.size() - size_t(flat[0]) - 1*/, start_score); // Report absolute

    size_t runs = 0; // count
    clock_t start_ts = clock();
    std::vector<dtype> winner = flat;

    // Start limiter high to allow some loops
    double limiter = pow(opt_frames_limiter, -min_opt_runs);

    double last_save = start_ts;
    double save_interval = CLOCKS_PER_SEC * 10;
    double last_reported = start_score;

    size_t max_non_improve = winner.size() * 2 + 1;
    size_t has_non_improve = 0;

    size_t p = 0;

    // Do a maximum amount of loops (bail out when needed)
    for (runs = 0; runs < opt_max_runs; runs += 1, p += 1)
    {
        // double best = INVSCORE;
        // std::vector<dtype> current = winner;

        if (p == ps.size()) p = 0;

        double best = GuessFrameParameters(data, winner, ps[p], 
            guess_params_e_from, guess_params_e_to, guess_params_e_step);

        clock_t now_ts = clock();
        if (now_ts > last_save + save_interval)
        {
            printf("Optimized by %.4g%% (-%.5g) {%.6g} (%zd runs in %.2fs) %g\n",
                (start_score - score) / start_score * 100.0, last_reported - score, score,
                runs + 1, (double(clock()) - start_ts) / CLOCKS_PER_SEC,
                limiter);
            std::vector<FrameTerms> frames = FlatToFrames(winner);
            std::vector<dtype> check = FramesToFlat(frames);
            if (check != winner) throw new std::exception("error");
            SaveFrames(frames, path);
            last_reported = score;
            last_save = now_ts;
        }

        if (best >= score) {
            has_non_improve++;
            if (has_non_improve > max_non_improve) {
                // printf("Too many non improves\n");
                break;
            }
        }
        else
        {
            has_non_improve = 0;

            // Sum up all optimizations 
            limiter = (limiter + (score - best) / score) * opt_limiter;
            //printf("  optimized by %g\n", score - best);
            // Update best score
            // winner = current;
            score = best;
            // Bail out when limit is reached
            if (limiter < opt_limit) {
                // printf("Limiter reached\n");
                break;
            }

        }



    }

    printf("=========================================================================\n");
    printf("Optimized by %.4g%% (-%.5g) {%.6g} (%zd runs in %.2fs)\n",
        (start_score - score) / start_score * 100.0, start_score - score, score,
        runs, (double(clock()) - start_ts) / CLOCKS_PER_SEC);
    printf("=========================================================================\n");

    std::vector<FrameTerms> frames = FlatToFrames(winner);
    SaveFrames(frames, path);

    return winner;
}

DataPoints GetDeltaPoints(const DataPoints& data, const FrameTerms& terms)
{
    DataPoints delta = data; // copy
    for (size_t p = 0; p < delta.size(); p += 1)
    {
        double ts = double(p) / (delta.size() - 1);
        for (size_t t = 0; t < terms.size() - 2; t += 3)
        {
            delta[p] -= sin((ts * terms[t + 0]
                + terms[t + 1])) * terms[t + 2];
        }
    }
    return delta;
}

bool FitFrameWave(const DataPoints& data, double max_amp32, FrameTerms& terms, double cuting)
{
    clock_t search_ts = clock();
    // Create copy and subtract current terms
    DataPoints delta = GetDeltaPoints(data, terms);
    
    dtype fmin = *std::min_element(data.begin(), data.end());
    dtype fmax = *std::max_element(data.begin(), data.end());
    double fmax_amp = std::max(abs(fmin), abs(fmax));

    dtype dmin = *std::min_element(delta.begin(), delta.end());
    dtype dmax = *std::max_element(delta.begin(), delta.end());
    double dmax_amp = std::max(abs(dmin), abs(dmax));

    // Only check score after optimizing
    double start_score = GetFrameScore(delta, { 0 });
    // Analyze rest data and try to figure out next frequency
    printf("Max amp %g vs frame %g vs delta %g\n", max_amp32, fmax_amp, dmax_amp);

    std::vector<dtype> winner = GuessNextFrameFrequency(delta, dmax_amp / fmax_amp, cuting);
    // double return_score = GetFrameScore(delta, winner);

    // Make copy of this frame
    FrameTerms test(terms);
    // Insert winner into test
    test.insert(test.begin(),
        winner.begin(), winner.end());

    double self_score = GetFrameScore(data, winner);
    double test_score = GetFrameScore(data, test);
    printf(" %02zd) [%.4g] <%.4g> sin(%.12g + %.12g) * %.12g (in %.2fs)\n",
        (terms.size() + 1) / 3, test_score, self_score,
        winner[0] / TAU, winner[1] / TAU, winner[2],
        (double(clock()) - search_ts) / CLOCKS_PER_SEC);

    // Try to optimize the whole thing
    test = OptimizeFrameTerms(data, test, 
        opt_frame_max_runs, opt_frame_limiter, opt_frame_limit);
    // Only check score after optimizing
    double best = GetFrameScore(data, test);
    // Update original frame if better
    if (best < start_score) terms = test;
    // Return the better score
    return best < start_score;
}

bool FitFrameWaves(const DataPoints& data, std::vector<FrameTerms>& frames,
    double max_amp, const std::string& path)
{
    bool had_better = false;


    // Send result to disk
    SaveFrames(frames, path);
    if (frames.empty()) throw new std::invalid_argument("frames");

    std::vector<dtype> flat1 = FramesToFlat(frames);
    double start_score = GetFramesScore(data, flat1);

    while ((frames.front().size() - 2) / 3 < max_frame_terms)
    {
        // dtype sum = std::reduce(data.begin(), data.end());
        // printf("Flat sum to fit frame wave(s) was %g\n", sum);
        // Try to add a new wave to frame terms
        bool first = frames.size() == 0 || frames.size() == 1
            && frames.front().size() < 3;
        bool better = FitFrameWave(data, max_amp, frames.front(),
            first ? 0.5 : 0.75);

        std::vector<dtype> flat2 = FramesToFlat(frames);
        double best_score = GetFramesScore(data, flat2);
        printf("Frames improved by %.4f%% (%.6g => %.6g)\n",
            100.0 - best_score / start_score * 100.0,
            start_score, best_score);

        // True if once true
        had_better |= better;
        // Abort if not improved
        if (better == false) break;
        // Send result to disk
        SaveFrames(frames, path);
        // Always fully optimize?
        // This is very expensive!
        if (optimize_all) break;
    }
    return had_better;
}
/*
void FitNewFrame(std::vector<dtype>& data, const std::string& path)
{
    // Get data range for the current frame to fit new waves to
    dtype dmin = *std::min_element(data.begin(), data.end());
    dtype dmax = *std::max_element(data.begin(), data.end());
    // Report the frame dimension before normalizing
    printf(" Frame min: %.32g\n", dmin);
    printf(" Frame max: %.32g\n", dmax);
    // Bring the data range to a uniform -1 to +1 range
    // This will help by translating the average to zero
    interp(data.begin(), data.end(), dmin, dmax, -1, 1);
    // Prepare a new frame
    FrameTerms terms;
    terms.push_back(dmin);
    terms.push_back(dmax);
    // Call the next function
    double score = GetFrameScore(data, terms);
    double best = FitFrameWaves(data, terms, path, score);
}
*/
template <typename T>
dtype EvalFrames(double ts, T begin, T end)
{
    double value = 0.0;
    while (begin < end)
    {
        const FrameTerms& frame = *begin;
        size_t terms = frame.size();
        for (size_t t = 0; t < terms - 2; t += 3)
        {
            value += sin((ts * frame[t + 0]
                + frame[t + 1])) * frame[t + 2];
        }
        double min = frame[terms - 2], max = frame[terms - 1];
        value = min + (value + 1) / 2 * (max - min);
        begin += 1;
    }
    return value;
}

void FitOldFrames(const std::vector<dtype>& original, const std::string& path)
{
    std::vector<dtype> data(original); // keep original
    // Load previous frames and terms (ToDo: if it exists)
    std::vector<FrameTerms> frames = LoadFrames(path);

    // Get data range for the current frame to fit new waves to
    dtype omin = *std::min_element(original.begin(), original.end());
    dtype omax = *std::max_element(original.begin(), original.end());
    double oamp = std::max(abs(omin), abs(omax));
    double org_range = omax - omin;

    // Curent frame min and max values
    dtype fmin = omin, fmax = omax;

    // Create first frame if necessary
    if (frames.empty())
    {
        // Get data range for the current frame to fit new waves to
        dtype min = *std::min_element(data.begin(), data.end());
        dtype max = *std::max_element(data.begin(), data.end());
        // Report the frame dimension before normalizing
        printf("Creating first frame, interpolating ...\n");
        printf(" min: %.32g\n", min);
        printf(" max: %.32g\n", max);
        interp(data.begin(), data.end(), min, max, -1, 1);
        frames.push_back({ min, max });
        fmin = min, fmax = max;
    }
    else
    {
        // Apply all steps as with original code
        // ToDo: might be able to unroll a bit of code
        for (size_t f = frames.size() - 1; f != 0; f -= 1)
        {
            size_t terms = (frames[f].size() - 2);
            double min = frames[f][frames[f].size() - 2];
            double max = frames[f][frames[f].size() - 1];
            // Report the frame dimension before normalizing
            printf("Loaded frame with %zd wave(s), interpolating ...\n", terms / 3);
            printf(" min: %.32g\n", min);
            printf(" max: %.32g\n", max);
            interp(data.begin(), data.end(), min, max, -1, 1);
            fmin = min, fmax = max;
            for (size_t t = 0; t < terms; t += 3)
            {
                printf(" %02zd) sin(t * %.12g + %.12g) * %.12g\n",
                    t / 3 + 1, // winner.score / caching.points * 100.0,
                    frames[f][t + 0] / TAU, frames[f][t + 1] / TAU, frames[f][t + 2]);
            }
            // First apply all previous frames, so we can start over
            for (size_t p = 0; p < data.size(); p += 1)
            {
                double ts = double(p) / (data.size() - 1);
                for (size_t t = 0; t < terms; t += 3)
                {
                    data[p] -= sin((ts * frames[f][t + 0]
                        + frames[f][t + 1])) * frames[f][t + 2];
                }
            }
        }

        if (frames.size() > 0)
        {
            size_t terms = (frames[0].size() - 2);
            double min = frames[0][frames[0].size() - 2];
            double max = frames[0][frames[0].size() - 1];
            // Report the frame dimension before normalizing
            for (size_t t = 0; t < terms; t += 3)
            {
                printf(" %02zd) sin(t * %.12g + %.12g) * %.12g\n",
                    t / 3 + 1, // winner.score / caching.points * 100.0,
                    frames[0][t + 0] / TAU, frames[0][t + 1] / TAU, frames[0][t + 2]);
            }

            printf("Interpolating current frame ...\n");
            printf(" min: %.32g\n", min);
            printf(" max: %.32g\n", max);
            interp(data.begin(), data.end(), min, max, -1, 1);
            fmin = min, fmax = max;
        }

        if (optimize_on_start)
        {
            // Optimize the full frame now
            std::vector<dtype> flat = FramesToFlat(frames);
            double max_amp = std::max(abs(fmin), abs(fmax));
            flat = OptimizeFrames(original, flat, max_amp, path,
                opt_load_max_runs, opt_load_limiter, opt_load_limit);
            // score = GetFramesScore(original, flat);
            frames = FlatToFrames(flat);
        }

    }

    std::vector<dtype> flat1 = FramesToFlat(frames);
    double org_score = GetFramesScore(original, flat1);

    while (true)
    {
        // FrameTerms& terms = frames.front();
        // Load previous terms if available
        // ToDo: make this more "dynamic"
        // Try to fit more frame waves
        //double score32 = GetFramesScore(original, FramesToFlat(frames));
        //double score24 = GetFrameScore(data, frames.front());
        //printf("Fitting new wave starting with avg error of %g (%g)\n", score32, score32 * data.size()); // full error
        //printf("Fitting new wave starting with avg error of %g (%g)\n", score24, score24 * data.size()); // normalized

        std::vector<dtype> flat1 = FramesToFlat(frames);
        double start_score = GetFramesScore(original, flat1);

        double max_amp = std::max(abs(fmin), abs(fmax));
        bool had_better = FitFrameWaves(data, frames, max_amp, path);

        // printf("The Fitter reported score %g\n", best);
        // Abort if nothing was added
        if (frames.front().size() == 2) break;

        // Send result to disk
        SaveFrames(frames, path);

        // Optimize the full frame now

        std::vector<dtype> flat = FramesToFlat(frames);
        flat = OptimizeFrames(original, flat, max_amp, path,
            opt_frames_max_runs, opt_frames_limiter, opt_frames_limit);
        // score = GetFramesScore(original, flat);

        double opt_score = GetFramesScore(original, flat);

        frames = FlatToFrames(flat);

        printf("Frame improved by %.4f%% (%.6g => %.6g)\n",
            100.0 - opt_score / start_score * 100.0,
            start_score, opt_score);

        printf("Last optimization %g\n", 1.0 - opt_score / start_score);

        if (opt_score / org_score < abort_err)
        {
            printf("Finished frame fitting, result is satisfying, goodbye!\n");
            // Send result to disk
            SaveFrames(frames, path);
            // exit(1);
        }

        if (1.0 - opt_score / start_score < abort_err * org_range)
        {
            printf("Finished frame fitting, no more improvements, goodbye!\n");
            // Send result to disk
            SaveFrames(frames, path);
            // exit(1);
        }

        if (opt_score >= start_score)
        {
            printf("Abort Abort, score has regressed even when optimized\n");
        }

        // Nothing improved
        if (had_better == false || (frames.front().size() - 2) / 3 > max_waves_per_frame)
        {


            std::vector<dtype> flat = FramesToFlat(frames);
            printf("Fzck %g\n", GetFramesScore(original, flat));

            // Get data range for the current frame to fit new waves to
            dtype dmin2 = *std::min_element(data.begin(), data.end());
            dtype dmax2 = *std::max_element(data.begin(), data.end());
            //
            //frames.front()[frames.front().size() - 2] = dmin2;
            //frames.front()[frames.front().size() - 1] = dmax2;

            // Need to recreate data?
            data = original; // copy again

                    // Apply all steps as with original code
            // ToDo: might be able to unroll a bit of code
            for (size_t f = frames.size() - 1; f != std::string::npos; f -= 1)
            {
                size_t terms = (frames[f].size() - 2);
                double min = frames[f][frames[f].size() - 2];
                double max = frames[f][frames[f].size() - 1];
                // Report the frame dimension before normalizing
                printf("Recreate frame with %zd wave(s), interpolating ...\n", terms / 3);
                printf(" min: %.32g\n", min);
                printf(" max: %.32g\n", max);
                interp(data.begin(), data.end(), min, max, -1, 1);
                // for (size_t t = 0; t < terms; t += 3)
                // {
                //     printf(" %02zd) sin(t * %.12g + %.12g) * %.12g\n",
                //         t / 3 + 1, // winner.score / caching.points * 100.0,
                //         frames[f][t + 0] / TAU, frames[f][t + 1] / TAU, frames[f][t + 2]);
                // }
                // First apply all previous frames, so we can start over
                for (size_t p = 0; p < data.size(); p += 1)
                {
                    double ts = double(p) / (data.size() - 1);
                    for (size_t t = 0; t < terms; t += 3)
                    {
                        data[p] -= sin((ts * frames[f][t + 0]
                            + frames[f][t + 1])) * frames[f][t + 2];
                    }
                }

            }

            //double min = frames[f][frames[f].size() - 2];
            //double max = frames[f][frames[f].size() - 1];
            //// Report the frame dimension before normalizing
            //printf("Apply frame with %zd wave(s), interpolating ...\n", terms / 3);
            //printf(" min: %.32g\n", min);
            //printf(" max: %.32g\n", max);
            //interp(data.begin(), data.end(), min, max, -1, 1);
            //for (size_t t = 0; t < terms; t += 3)
            //{
            //    printf(" %02zd) sin(t * %.12g + %.12g) * %.12g\n",
            //        t / 3 + 1, // winner.score / caching.points * 100.0,
            //        frames[f][t + 0] / TAU, frames[f][t + 1] / TAU, frames[f][t + 2]);
            //}




            // remove the current frame from data

            // Get data range for the current frame to fit new waves to
            dtype dmin = *std::min_element(data.begin(), data.end());
            dtype dmax = *std::max_element(data.begin(), data.end());
            // Insert another frame and record the min and max values

            // Should have no influence if no terms there
            frames.insert(frames.begin(), { dmin, dmax });
            // Report the frame dimension before normalizing
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            if (!had_better) printf("No more waves found to apply, reframing ...\n");
            else printf("Limit for waves per frame reached, reframing ...\n");
            // printf(" Applying %zd wave(s) to frame and adding new one.\n", terms / 3);
            printf(" min: %.32g\n", dmin);
            printf(" max: %.32g\n", dmax);
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            // Bring the data range to a uniform -1 to +1 range
            // This will help by translating the average to zero
            interp(data.begin(), data.end(), dmin, dmax, -1, 1);

            std::vector<dtype> flat2 = FramesToFlat(frames);
            printf("Fzck %g\n", GetFramesScore(original, flat2));

            printf("Fzck %g\n", GetFrameScore(data, {0, 0}));

            // Send result to disk
            SaveFrames(frames, path);

            continue;
        }



        // Need to recreate data?
        data = original; // copy again

                // Apply all steps as with original code
        // ToDo: might be able to unroll a bit of code
        for (size_t f = frames.size() - 1; f != 0; f -= 1)
        {
            size_t terms = (frames[f].size() - 2);
            double min = frames[f][frames[f].size() - 2];
            double max = frames[f][frames[f].size() - 1];
            // Report the frame dimension before normalizing
            printf("Recreate frame with %zd wave(s), interpolating ...\n", terms / 3);
            printf(" min: %.32g\n", min);
            printf(" max: %.32g\n", max);
            interp(data.begin(), data.end(), min, max, -1, 1);
            // for (size_t t = 0; t < terms; t += 3)
            // {
            //     printf(" %02zd) sin(t * %.12g + %.12g) * %.12g\n",
            //         t / 3 + 1, // winner.score / caching.points * 100.0,
            //         frames[f][t + 0] / TAU, frames[f][t + 1] / TAU, frames[f][t + 2]);
            // }
            // First apply all previous frames, so we can start over
            for (size_t p = 0; p < data.size(); p += 1)
            {
                double ts = double(p) / (data.size() - 1);
                for (size_t t = 0; t < terms; t += 3)
                {
                    data[p] -= sin((ts * frames[f][t + 0]
                        + frames[f][t + 1])) * frames[f][t + 2];
                }
            }

        }

        if (frames.size() > 0)
        {
            size_t terms = (frames[0].size() - 2);
            double min = frames[0][frames[0].size() - 2];
            double max = frames[0][frames[0].size() - 1];
            // Report the frame dimension before normalizing
            printf("Interpolating current frame ...\n");
            printf(" min: %.32g\n", min);
            printf(" max: %.32g\n", max);
            interp(data.begin(), data.end(), min, max, -1, 1);
        }

        // Abort condition
        //if (best >= score) break;

        // Send result to disk
        SaveFrames(frames, path);

        // exit(3);



        
        // Add another frame 

    }

    printf("ABorted search\n");

}

int main(int argc, char* argv[])
{

    DWORD dwError, dwPriClass;

    if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN))
    {
        dwError = GetLastError();
        if (ERROR_PROCESS_MODE_ALREADY_BACKGROUND == dwError)
            _tprintf(TEXT("Already in background mode\n"));
        else _tprintf(TEXT("Failed to enter background mode (%d)\n"), dwError);
        exit(1);
    } 
    struct CmdGetOpt getopt;

    /*
    getopt.options.emplace_back(CmdOption{ 'f', "frequencies",
        "Number of frequencies to try per fourier step.", false, "16", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            sscanf_s(value.string, "%zu", &test_freqs); } });

    getopt.options.emplace_back(CmdOption{ 'r', "fitter-max-runs",
        "Maximum iterations for CPU fitting.", false, "750", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            opt_max_runs = atoi(value.string); } });

    getopt.options.emplace_back(CmdOption{ 'e', "rough-fit",
        "Number of exponents to try strictly.", false, "6", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            sscanf_s(value.string, "%zu", &rough_fit); } });

    getopt.options.emplace_back(CmdOption{ 'l', "fitter-limit",
        "Optimize until this limit is reached.", false, "0.001", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            sscanf_s(value.string, "%lf", &opt_limit); } });

    getopt.options.emplace_back(CmdOption{ 'm', "fitter-scale",
        "Scale to detect limit while fitting (0-1).", false, "0.5", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            sscanf_s(value.string, "%lf", &opt_limiter); } });


    getopt.options.emplace_back(CmdOption{ 's', "gpu-samples",
        "Number of GPU samples to try per parameter.", false, "0", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            SAMPLES = atoi(value.string); } });
    */

    getopt.options.emplace_back(CmdOption{ 'g', "gpu-device",
        "Device number of GPU to use for cuda.", false, "0", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            device = atoi(value.string); } });

    // Additional option parser target (just call me)
    getopt.options.emplace_back(CmdOption{ 'h', "help",
        "Show basic usage information.",
        false, NULL, false, NULL,
        getopt_help_cmd });

    // Now parse all passed arguments
    for (int i = 1; i < argc; i += 1) {
        getopt_parse(&getopt, argv[i]);
    }

    if (getopt.args.size() != 1)
    {
        err("Exepected exactly one arguments, got %d",
            getopt.args.size());
        exit(1);
    }

    // ToDo: check device count
    cudaSetDevice(device);

    // Make sync calls blocking to not waste any CPU
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // 1) Create new fit from presented data
    // 2a) Continue to fit the last frame
    // 2b) Continue to fit the next frame
    // 3) Optimize all previous frames/terms

    std::string input(getopt.args[0]);
    std::vector<dtype> data = LoadArray(input);

    printf("#########################################################################\n");
    printf("Loaded %s with %zd points\n", input.c_str(), data.size());
    printf("#########################################################################\n");

    FitOldFrames(data, input + ".terms");
    //FitNewFrame(data, input + ".terms");

    /*
    if (refit)
    {

        double* terms; double* data;
        int size = LoadData(file.c_str(), terms);
        std::vector params(terms, terms + size);
        int points = LoadData(getopt.args[0].c_str(), data);
        printf("Refitting %d %g %g\n", size, GetScore(data, points, params), data[0]);

        OptimizeAllParams(data, points, params, opt_max_runs,
            opt_limiter, opt_limit, getopt.args[0].c_str());


    }
    else
    {
        ProcessFile(getopt.args[0],
            test_freqs, trim_short, trim_long, rough_fit, SAMPLES,
            opt_max_runs, opt_limiter, opt_limit, cont);
    }
    */

    return 0;
}
