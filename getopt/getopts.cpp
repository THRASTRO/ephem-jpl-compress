#include <bitset>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "getopts.hpp"

// Backported my own code from libsass (2022 Marcel Greter) 

const std::bitset<256> tblWhitespace(
    "00000000000000000000000000000000" // 255 - 224
    "00000000000000000000000000000000" // 223 - 192
    "00000000000000000000000000000000" // 192 - 160
    "00000000000000000000000000000000" // 159 - 128
    "00000000000000000000000000000000" // 127 - 96
    "00000000000000000000000000000000" // 95 - 64
    "00000000000000000000000000000001" // 63 - 32
    "00000000000000000011111000000000" // 31 - 0
);

// Returns whether [character] is an ASCII whitespace character.
inline bool isWhitespace(uint8_t character) {
    return tblWhitespace[character];
}

// Trim the left side of passed string.
void makeLeftTrimmed(std::string& str) {
    if (str.begin() != str.end()) {
        auto pos = std::find_if_not(
            str.begin(), str.end(),
            isWhitespace);
        str.erase(str.begin(), pos);
    }
}
// EO makeLeftTrimmed

// Trim the right side of passed string.
void makeRightTrimmed(std::string& str) {
    if (str.begin() != str.end()) {
        auto pos = std::find_if_not(
            str.rbegin(), str.rend(),
            isWhitespace);
        str.erase(str.rend() - pos);
    }
}
// EO makeRightTrimmed

// Make the passed string whitespace trimmed.
void makeTrimmed(std::string& str) {
    makeLeftTrimmed(str);
    makeRightTrimmed(str);
}
// EO makeTrimmed

// Replace all occurrences of `search` in string `str` with `replacement`.
void makeReplace(std::string& str, const std::string& search, const std::string& replacement)
{
    size_t pos = str.find(search);
    while (pos != std::string::npos)
    {
        str.replace(pos, search.size(), replacement);
        pos = str.find(search, pos + replacement.size());
    }
}


// Optimized version where we know one side is already lowercase
bool _equalsIgnoreCaseConst(const char a, const char b) {
    return a == b || a == tolower(b);
}

// The difference between upper- and lowercase ASCII letters.
// `0b100000` can be bitwise-ORed with uppercase ASCII letters
// to get their lowercase equivalents.
const uint8_t asciiCaseBit = 0x20;

// Returns whether [character1] and [character2] are the same, modulo ASCII case.
bool characterEqualsIgnoreCase(uint8_t character1, uint8_t character2)
{
    if (character1 == character2) return true;

    // If this check fails, the characters are definitely different. If it
    // succeeds *and* either character is an ASCII letter, they're equivalent.
    if ((character1 ^ character2) != asciiCaseBit) return false;

    // Now we just need to verify that one of the characters is an ASCII letter.
    uint8_t upperCase1 = character1 & ~asciiCaseBit;
    return upperCase1 >= 'A' && upperCase1 <= 'Z';
}

bool _equalsIgnoreCase(const char a, const char b) {
    return characterEqualsIgnoreCase(a, b);
}

bool startsWithIgnoreCase(const std::string& str, const char* prefix, size_t len) {
    return len <= str.size() && std::equal(prefix, prefix + len, str.begin(), _equalsIgnoreCaseConst);
}

bool startsWithIgnoreCase(const std::string& str, const std::string& prefix) {
    return prefix.size() <= str.size() && std::equal(prefix.begin(), prefix.end(), str.begin(), _equalsIgnoreCase);
}

// Return joined string from all passed strings, delimited by separator.
std::string join(const std::vector<std::string>& strings, const char* separator)
{
    switch (strings.size())
    {
    case 0:
        return "";
    case 1:
        return strings[0];
    default:
        size_t size = strings[0].size();
        size_t sep_len = ::strlen(separator);
        for (size_t i = 1; i < strings.size(); i++) {
            size += sep_len + strings[i].size();
        }
        std::string os;
        os.reserve(size);
        os += strings[0];
        for (size_t i = 1; i < strings.size(); i++) {
            os += separator;
            os += strings[i];
        }
        return os;
    }
}
// EO join

std::string format_option(struct CmdGetOpt* getopt, CmdOption& option)
{
    std::stringstream line;
    if (option.shrt) {
        line << "-" << option.shrt;
        if (option.name) line << ", ";
    }
    else {
        line << "    ";
    }
    if (option.name) {
        line << "--";
        if (option.boolean) {
            line << "[no-]";
        }
        line << option.name;
    }
    if (option.argument) {
        if (option.optional) line << "[";
        line << "=";
        line << option.argument;
        if (option.optional) line << "]";
    }
    return line.str();
}

// Count number of printable bytes/characters
size_t count_printable(const char* string)
{
    size_t count = 0;
    while (string && *string) {
        if (string[0] == '\x1b' && string[1] == '[') {
            while (*string != 0 && *string != 'm') {
                string++;
            }
            string++;
        }
        else {
            string += 1;
            count += 1;
        }
    }
    return count;
}
// EO count_printable

void getopt_error(struct CmdGetOpt* getopt, const char* msg)
{
    printf(msg);
    printf("\n");
}

// Check for pending required option
void getopt_check_required_option(struct CmdGetOpt* getopt)
{
    // Expected argument, error
    if (getopt->needsArgument) {
        std::stringstream strm;
        if (getopt->needsArgumentWasShort) {
            std::string value(1, getopt->needsArgument->shrt);
            // StringUtils::makeReplace(value, "'", "\\'"); // only a char
            strm << "option '-" << value << "' requires an argument'";
        }
        else {
            std::string value(getopt->needsArgument->name);
            makeReplace(value, "'", "\\'");
            strm << "option '--" << value << "' requires an argument'";
        }
        std::string msg(strm.str());
        getopt_error(getopt, msg.c_str());
        return; // return after error
    }
}


std::vector<const CmdOption*> find_long_options(struct CmdGetOpt* getopt, const std::string& arg)
{
    std::vector<const CmdOption*> matches;
    for (const CmdOption& option : getopt->options) {
        if (startsWithIgnoreCase(option.name, arg)) {
            if (arg == option.name) return { &option };
            matches.push_back(&option);
        }
        if (option.boolean) {
            if (startsWithIgnoreCase(arg, "no-", 3)) {
                std::string name(arg.begin() + 3, arg.end());
                if (startsWithIgnoreCase(option.name, name)) {
                    if (arg == option.name) return { &option };
                    matches.push_back(&option);
                }
            }
        }
    }
    return matches;
}

std::vector<const CmdOption*> find_short_options(struct CmdGetOpt* getopt, const char arg)
{
    std::vector<const CmdOption*> matches;
    for (CmdOption& option : getopt->options) {
        if (option.shrt == arg) {
            matches.push_back(&option);
        }
    }
    return matches;
}

std::vector<const struct CmdGetOptEnum*> find_options_enum(
    const struct CmdGetOptEnum* enums, const std::string& arg)
{
    std::vector<const struct CmdGetOptEnum*> matches;
    while (enums && enums->string) {
        if (startsWithIgnoreCase(enums->string, arg)) {
            matches.push_back(enums);
        }
        enums += 1;
    }
    return matches;
}

// Function that must be consecutively called for every argument.
// Ensures to properly handle cases where a mandatory or optional
// argument, if required by the previous option, is handled correctly.
// This is a bit different to "official" GNU GetOpt, but should be
// reasonably well and support more advanced usages than before.
void getopt_parse(struct CmdGetOpt* getopt, const char* value)
{
    if (value == nullptr) return;
    std::string arg(value);
    makeTrimmed(arg);
    union CmdOptionValue result {};

    if (arg != "-" && arg != "--" &&
        arg[0] == '-' && getopt->wasAssignment.empty())
    {
        getopt_check_required_option(getopt);
        std::vector<const CmdOption*> opts;

        // Check if we have some assignment
        size_t assign = arg.find_first_of('=');
        if (assign != std::string::npos) {
            std::string key(arg.begin(), arg.begin() + assign);
            std::string val(arg.begin() + assign + 1, arg.end());
            getopt_parse(getopt, key.c_str());
            getopt->wasAssignment = key;
            getopt_parse(getopt, val.c_str());
            getopt->wasAssignment.clear();
            return;
        }

        // Long argument
        if (arg[1] == '-') {
            arg.erase(0, 2);
            opts = find_long_options(getopt, arg);
            getopt_check_required_option(getopt);
        }
        // Short argument
        else {
            arg.erase(0, 1);
            // Split multiple short args
            if (arg.size() > 1) {
                for (size_t i = 0; i < arg.size(); i += 1) {
                    std::string split("-"); split += arg[i];
                    getopt_parse(getopt, split.c_str());
                    getopt_check_required_option(getopt);
                    // break on first error
                }
                return;
            }
            opts = find_short_options(getopt, arg[0]);
            // Consume further until has arg
        }
        if (opts.size() == 0) {
            std::stringstream strm;
            strm << "unrecognized option '--" << arg << "'";
            std::string msg(strm.str());
            getopt_error(getopt, msg.c_str());
            return; // return after error
        }
        if (opts.size() > 1) {
            std::stringstream strm;
            if (value[0] == '-' && value[1] == '-') {
                strm << "option '--" << arg << "' is ambiguous; possibilities: ";
                for (auto opt : opts) strm << "'--" << opt->name << "'" << std::setw(4);
            }
            else {
                // Should never happen if you configured your options right
                strm << "option '-" << arg << "' is ambiguous1 (internal error)";
                for (auto opt : opts) strm << "'--" << opt->name << "'" << std::setw(4);
            }
            std::string msg(strm.str());
            getopt_error(getopt, msg.c_str());
            return; // return after error
        }
        getopt->lastArg = opts[0];
        getopt->needsArgument = opts[0]->argument ? opts[0] : nullptr;
        getopt->needsArgumentWasShort = value[0] == '-' && value[1] != '-';

        // Check boolean options right away
        if (opts[0]->boolean) {
            // Get boolean result (invert if argument has "no-" prefix)
            result.boolean = !startsWithIgnoreCase(arg, "no-", 3);
        }
        if (!getopt->needsArgument) {
            opts[0]->cb(getopt, result);
        }
    }
    else if (getopt->needsArgument) {
        if (getopt->needsArgument->enums) {
            auto matches = find_options_enum(
                getopt->needsArgument->enums, arg);
            if (matches.size() == 0) {
                std::stringstream strm;
                strm << "enum '" << arg << "' is not valid for option '";
                if (getopt->needsArgumentWasShort) {
                    strm << "-" << getopt->needsArgument->shrt;
                }
                else {
                    strm << "--" << getopt->needsArgument->name;
                }
                strm << "' (valid enums are ";
                auto enums = getopt->needsArgument->enums;
                std::vector<std::string> names;
                while (enums && enums->string) {
                    names.push_back(enums->string);
                    enums += 1;
                }
                strm << join(names, "or") << ")";
                std::string msg(strm.str());
                getopt_error(getopt, msg.c_str());
                return; // return after error
            }
            else if (matches.size() > 1) {

                std::stringstream strm;
                strm << "enum '" << arg << "' for option '";
                if (getopt->needsArgumentWasShort) {
                    strm << "-" << getopt->needsArgument->shrt;
                }
                else {
                    strm << "--" << getopt->needsArgument->name;
                }
                strm << "' is ambiguous (possibilities are ";
                std::vector<std::string> names;
                for (auto match : matches) {
                    names.push_back(match->string);
                }
                strm << join(names, "or") << ")";
                std::string msg(strm.str());
                getopt_error(getopt, msg.c_str());
                return; // return after error
            }
            result.integer = matches[0]->enumid;
        }
        else {
            result.string = arg.c_str();
        }
        getopt->needsArgument->cb(getopt, result);
        getopt->needsArgumentWasShort = false;
        getopt->needsArgument = nullptr;
    }
    else if (!getopt->wasAssignment.empty()) {
        std::stringstream strm;
        strm << "option '";
        if (getopt->lastArgWasShort) {
            strm << "-" << getopt->lastArg->shrt;
        }
        else {
            strm << "--" << getopt->lastArg->name;
        }
        strm << "' doesn't allow an argument";
        std::string msg(strm.str());
        getopt_error(getopt, msg.c_str());
        return; // return after error
    }
    else {
        // This is a regular argument
        getopt->args.push_back(arg);
    }

}

void getopt_print_help(struct CmdGetOpt* getopt, std::ostream& stream)
{
    size_t longest = 20;
    // Determine longest option to align them all
    for (CmdOption& option : getopt->options) {
        std::string fmt(format_option(getopt, option));
        size_t len = count_printable(fmt.c_str()) + 2;
        if (len > longest) longest = len;
    }
    // Print out each option line by line
    for (CmdOption& option : getopt->options) {
        std::string fmt(format_option(getopt, option));
        size_t len = count_printable(fmt.c_str());
        stream << "  " << fmt;
        stream.width(longest - len + 2);
        stream << " " << option.desc;
        stream << "\n";
        if (option.enums) {
            auto enums = option.enums;
            std::vector<std::string> names;
            while (enums && enums->string) {
                names.push_back(enums->string);
                enums += 1;
            }
            stream << std::setw(longest + 4) << "";
            if (option.argument) {
                stream << option.argument;
                stream << " must be ";
            }
            stream << join(names, "or") << "\n";
        }
    }
}
