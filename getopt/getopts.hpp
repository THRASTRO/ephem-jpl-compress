#pragma once

#include <string>
#include <vector>

// Struct must be known in order to access it in the callback
// We don't expect this to change much once it is proven good
// We also don't want to support all cases under the sun!
union CmdOptionValue {
    int integer;
    bool boolean;
    const char* string;
};

class CmdOption {
public:
    const char shrt = '\0';
    const char* name;
    const char* desc;
    const bool boolean = false;
    const char* argument = nullptr;
    const bool optional = false;
    const struct CmdGetOptEnum* enums;
    void (*cb) (struct CmdGetOpt* getopt, union CmdOptionValue value);
    CmdOption(
        const char shrt = '\0',
        const char* name = nullptr,
        const char* desc = nullptr,
        const bool boolean = false,
        const char* argument = nullptr,
        const bool optional = false,
        const struct CmdGetOptEnum* enums = nullptr,
        void (*cb) (struct CmdGetOpt* getopt, union CmdOptionValue value) = nullptr
    ) :
        shrt(shrt),
        name(name),
        desc(desc),
        boolean(boolean),
        argument(argument),
        optional(optional),
        enums(enums),
        cb(cb)
    {}
};

struct CmdGetOpt {
    std::string wasAssignment;
    bool lastArgWasShort = false;
    bool needsArgumentWasShort = false;
    const CmdOption* lastArg = nullptr;
    const CmdOption* needsArgument = nullptr;
    std::vector<std::string> args = {};
    std::vector<CmdOption> options;
    std::vector<struct CmdArgument> arguments;
    CmdGetOpt() {}
};

// Single enumeration item for sass options
// Maps an option to the given enum integer.
struct CmdGetOptEnum
{
public:
    int enumid;
    const char* string;
    CmdGetOptEnum(const char* name, int id) :
        enumid(id),
        string(name)
    {}
};

struct CmdArgument {
    bool optional = false;
    const char* name;
    void (*cb) (struct CmdGetOpt* getopt, const char* arg);
    CmdArgument(
        bool optional = false,
        const char* name = nullptr,
        void (*cb) (struct CmdGetOpt* getopt, const char* arg) = nullptr
    ) :
        optional(optional),
        name(name),
        cb(cb)
    {}
};

void getopt_parse(struct CmdGetOpt* getopt, const char* value);

void getopt_print_help(struct CmdGetOpt* getopt, std::ostream& stream);
