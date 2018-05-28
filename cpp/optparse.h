/*
 * Class 'optparse' implements a parser for GNU-style command-line arguments.
 * Inherit this class to define your own option variables and to implement an
 * option handler with macros, BEGIN_OPTION_MAP, ON_OPTION(_WITH_ARG), and
 * END_OPTION_MAP.
 */

#ifndef __OPTPRASE_H
#define __OPTPRASE_H

#include <cstring>
#include <stdexcept>


/**
 * An event-driven parser for command-line arguments.
 */
class optparse {
public:
  /**
   * Exception class for unrecognized options.
   */
  class unrecognized_option : public std::invalid_argument {
  public:
    explicit unrecognized_option(char shortopt)
        : std::invalid_argument(std::string("-") + shortopt) {}
    explicit unrecognized_option(const std::string& longopt)
        : std::invalid_argument(std::string("--") + longopt) {}
  };
  /**
   * Exception class for invalid values.
   */
  class invalid_value : public std::invalid_argument {
  public:
    explicit invalid_value(const std::string& message)
        : std::invalid_argument(message) {}
  };
  /**
   * Parse options.
   *  @param  argv        array of null-terminated strings to be parsed
   *  @param  num_argv    specifies the number, in strings, of the array
   *  @return             the number of used arguments
   *  @throws             optparse_exception
   */
  int parse(char* argv[], int num_argv)
  {
    int i;
    for (i = 1;i < num_argv;++i) {
      const char *token = argv[i];
      if (*token++ == '-') {
        const char *next_token = (i+1 < num_argv) ? argv[i+1] : "";
        if (*token == '\0') break;  // only '-' was found.
        if (*token == '-') {
          const char *arg = std::strchr(++token, '=');
          if (arg != nullptr) {
            ++arg;
          } else {
            arg = next_token;
          }
          int ret = handle_option(0, token, arg);
          if (ret < 0) {
            throw unrecognized_option(token);
          }
          if (arg == next_token) {
            i += ret;
          }
        } else {
          char c;
          while ((c = *token++) != '\0') {
            const char *arg = (*token != '\0') ? token : next_token;
            int ret = handle_option(c, "", arg);
            if (ret < 0) {
              throw unrecognized_option(c);
            }
            if (ret > 0) {
              if (arg == next_token) {
                i += ret;
              } else {
                token = "";
              }
            }
          } // while
        } // else (*token == '-')
      } else {
        break;  // a non-option argument was fonud.
      }
    } // for (i)

    return i;
  }

protected:
  /**
   * Option handler
   *  This function should be overridden by inheritance class.
   *  @param  c           short option character, 0 for long option
   *  @param  longname    long option name
   *  @param  arg         an argument for the option
   *  @return             0 (success);
                          1 (success with use of an argument);
                          -1 (failed, unrecognized option)
   *  @throws             option_parser_exception
   */
  virtual int handle_option(char c, const char *longname, const char *arg) = 0;

  int __optstrcmp(const char *option, const char *longname)
  {
    const char *p = std::strchr(option, '=');
    return (p != nullptr) ?
           std::strncmp(option, longname, p-option) :
           std::strcmp(option, longname);
  }
};


/** The begin of inline option map. */
#define BEGIN_OPTION_MAP() \
  int handle_option(char __c, const char *__longname, const char *arg) override { \
    int used_args = -1; \
    if (false) { \

/** An entry of option map */
#define ON_OPTION(test) \
    } else if (test) { \
      used_args = 0; \

#define ON_OPTION_WITH_ARG(test) \
    } else if (test) { \
      used_args = 1; \

/** The end of option map implementation */
#define END_OPTION_MAP() \
    } \
    return used_args; \
  } \

/** A predicator for short options */
#define SHORTOPT(x)     (__c == (x))
/** A predicator for long options */
#define LONGOPT(x)      (__optstrcmp(__longname, x) == 0)

#endif //__OPTPRASE_H
