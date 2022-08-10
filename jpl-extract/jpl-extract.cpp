#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#include <filesystem>
#include <sstream>
#include <fstream>
#include <format>

#include "SpiceUsr.h"
#include "getopts.hpp"

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

std::string prefix("");
std::string root("satellite");
size_t points = 51201;

int main(int argc, const char** argv)
{

    struct CmdGetOpt getopt;

    getopt.options.emplace_back(CmdOption{ 'n', "name",
        "Set output file prefix", false, "NAME", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            prefix = std::string(value.string) + "."; } });

    // Additional option parser target (just call me)
    getopt.options.emplace_back(CmdOption{ 'd', "root",
        "Set base export directory.", false, "PATH", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            root = std::string(value.string); } });

    // Additional option parser target (just call me)
    getopt.options.emplace_back(CmdOption{ 'p', "points",
        "Number of data points to export.", false, "51201", false, nullptr,
        [](struct CmdGetOpt* getopt, union CmdOptionValue value) {
            sscanf_s(value.string, "%zu", &points); } });

    // Additional option parser target (just call me)
    getopt.options.emplace_back(CmdOption{ 'h', "help",
        "Show basic usage information.",
        false, NULL, false, NULL,
        getopt_help_cmd });

    // Now parse all passed arguments
    for (int i = 1; i < argc; i += 1) {
        getopt_parse(&getopt, argv[i]);
    }

    if (getopt.args.size() != 2)
    {
        err("Exepected exactly 2 arguments");
        exit(1);
    }

    std::string path = getopt.args[0];
    int body = atoi(getopt.args[1].c_str());

    /*
  
    DAF/SPK file format notes:

    ic[0] - target code
    ic[1] - center code
    ic[2] - frame code
    ic[3] - representation code (2 == Chebyshev position only)
    ic[4] - initial address of array
    ic[5] - final address of array

    len = ic[5] - ic[4] + 1

    dc[0] - initial epoch of data (seconds relative to J2000)
    dc[1] - final epoch of data (seconds relative to J2000)

    */

    SpiceInt daf;
    SpiceBoolean found;
    union
    {
        SpiceDouble d[128];
        SpiceChar c[1024];
    } sum;


    // open the file and verifiy that it is a DAF SPK file
    dafopr_c(path.c_str(), &daf); // open SPK file for reading
    if (failed_c())
    {
        reset_c();
        err("could not open %s as a DAF", path.c_str());
        return NULL;
    }
    dafgsr_c(daf, 1, 1, 128, sum.d, &found); // read first record
    if (failed_c() || !found || memcmp(sum.c, "DAF/SPK ", 8))
    {
        reset_c();
        dafcls_c(daf);
        err("%s is not an SPK file", path.c_str());
        return NULL;
    }

    double start = INFINITY;
    double stop = -INFINITY;

    const SpiceInt nd = 2;
    const SpiceInt ni = 6;
    SpiceDouble dc[2];
    SpiceInt ic[6];

    dafbfs_c(daf); // begin forward search
    while (daffna_c(&found), found)
    {                                   // find the next array
        dafgs_c(sum.d);                 // get array summary
        dafus_c(sum.d, nd, ni, dc, ic); // unpack the array summary
        if (failed_c())
            break;
        if (ic[0] != body)
            continue;
        if (start == INFINITY)
        {
            start = dc[0];
            stop = dc[1];
        }
        else if (start == dc[1])
        {
            start = dc[0];
        }
        else if (stop == dc[0])
        {
            stop = dc[0];
        }
        else
        {
            err("Parts not connected");
            exit(1);
        }
    }

    printf("Starts: %.22g\n", start);
    printf("Stops: %.22g\n", stop);

    constexpr const char* sfmt = "{}/{}/states/{}{}.f64";
    int flags = std::ios::out | std::ios::trunc | std::ios::binary;

    std::filesystem::create_directory(std::format("{}/{}", root, body));
    std::filesystem::create_directory(std::format("{}/{}/states", root, body));
    std::filesystem::create_directory(std::format("{}/{}/orbitals", root, body));

    if (failed_c())
    {
        reset_c();
        dafcls_c(daf);
        err("no valid Chebyshev position-only segments in %s", path.c_str());
        return NULL;
    }

    SpiceInt eph;
    SpiceInt frame;
    SpiceInt center;
    SpiceChar id[41];
    SpiceDouble desc[5];
    SpiceDouble pv[6];

    furnsh_c(path.c_str());

    path = std::format(sfmt, root, body, prefix, "px"); std::ofstream fh_px(path, flags);
    if (!fh_px.is_open()) { err("Could not open %s", path.c_str()); exit(1); }
    path = std::format(sfmt, root, body, prefix, "py"); std::ofstream fh_py(path, flags);
    if (!fh_py.is_open()) { err("Could not open %s", path.c_str()); exit(1); }
    path = std::format(sfmt, root, body, prefix, "pz"); std::ofstream fh_pz(path, flags);
    if (!fh_pz.is_open()) { err("Could not open %s", path.c_str()); exit(1); }
    path = std::format(sfmt, root, body, prefix, "vx"); std::ofstream fh_vx(path, flags);
    if (!fh_vx.is_open()) { err("Could not open %s", path.c_str()); exit(1); }
    path = std::format(sfmt, root, body, prefix, "vy"); std::ofstream fh_vy(path, flags);
    if (!fh_vy.is_open()) { err("Could not open %s", path.c_str()); exit(1); }
    path = std::format(sfmt, root, body, prefix, "vz"); std::ofstream fh_vz(path, flags);
    if (!fh_vz.is_open()) { err("Could not open %s", path.c_str()); exit(1); }

    double spans = (stop - start) / points;
    for (double t = start; t <= stop; t += spans)
    {
        spksfs_c(body, t, sizeof(id), &eph, desc, id, &found);

        if (!found)
        {
            err("Body %d not found at %g!", body, t);
            exit(1);
        }
        spkpvn_c(eph, desc, t, &frame, pv, &center);
        fh_px.write(reinterpret_cast<char*>(&pv[0]), sizeof(double));
        fh_py.write(reinterpret_cast<char*>(&pv[1]), sizeof(double));
        fh_pz.write(reinterpret_cast<char*>(&pv[2]), sizeof(double));
        fh_vx.write(reinterpret_cast<char*>(&pv[3]), sizeof(double));
        fh_vy.write(reinterpret_cast<char*>(&pv[4]), sizeof(double));
        fh_vz.write(reinterpret_cast<char*>(&pv[5]), sizeof(double));
    }

}
