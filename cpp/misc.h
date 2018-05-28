#ifndef __MISC_H
#define __MISC_H

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <typeinfo>
#include <utility>
#include <initializer_list>
#include <cassert>

#ifdef DEBUG
constexpr bool DEBUG_TEST = true;
#else
constexpr bool DEBUG_TEST = false;
#endif
#define debug_print(...) \
  do { if (DEBUG_TEST) printf(__VA_ARGS__); } while (0)

namespace misc {

  std::vector<std::string> split(const std::string& s, char delim);

  std::string getField(const std::string& s, const std::string& desc, const std::string& lch, const std::string& rch);

  template <typename InputIter>
  std::string mkString(InputIter begin, InputIter end,
                       const std::string& l, const std::string& sep, const std::string& r) {
    std::stringstream ss;
    ss << std::scientific << l;
    if (begin != end) {
      InputIter cur = begin;
      ss << *cur;
      ++cur;
      while (cur != end) {
        ss << sep;
        ss << *cur;
        ++cur;
      }
    }
    ss << r;
    return ss.str();
  }

  bool isLittleEndian();

  template <typename T>
  std::string toBytes(T x, bool bytesOrder) {
    const void* p = &x;
    auto cp = static_cast<const char*>(p);
    std::string cast(cp, cp + sizeof(T));
    if (bytesOrder)
      return cast;
    else
      return std::string(cast.crbegin(), cast.crend());
  }

  template <typename T, typename InputInter>
  T fromBytes(InputInter begin, InputInter end, bool bytesOrder) {
    std::string cs;
    unsigned int sz = sizeof(T);
    cs.reserve(sz);
    InputInter cur = begin;
    for (unsigned int i = 0; i != sz; ++i) {
      if (cur != end) {
        cs.push_back(*cur);
        ++cur;
      } else {
        cs.push_back(static_cast<char>(0x00));
      }
    }
    const void* p;
    if (bytesOrder)
      p = cs.data();
    else
      p = std::string(cs.crbegin(), cs.crend()).data();
    auto cp = static_cast<const T*>(p);
    return *cp;
  };

  template <typename T>
  std::string numpy_dtype() {
    std::string ret;
    ret += isLittleEndian()? '<' : '>';
    const std::vector<std::pair<const std::type_info&, char>> map_type {
        {typeid(float), 'f'},
        {typeid(double), 'f'},
        {typeid(long double), 'f'},

        {typeid(int), 'i'},
        {typeid(char), 'i'},
        {typeid(short), 'i'},
        {typeid(long), 'i'},
        {typeid(long long), 'i'},

        {typeid(unsigned int), 'u'},
        {typeid(unsigned char), 'u'},
        {typeid(unsigned short), 'u'},
        {typeid(unsigned long), 'u'},
        {typeid(unsigned long long), 'u'},

        {typeid(bool), 'b'}
    };
    auto mtp = map_type.cbegin();
    for (; mtp != map_type.cend(); ++mtp) if (mtp->first == typeid(T)) break;
    ret += mtp->second;
    ret += std::to_string(static_cast<unsigned int>(sizeof(T)));
    return ret;
  }

  template <typename T>
  std::string createNpyHeader(bool fortran_order, std::initializer_list<unsigned int> ds) {
    std::string dict;
    dict += "{'descr': '";
    dict += numpy_dtype<T>();
    dict += "', 'fortran_order': ";
    dict += fortran_order? "True" : "False";
    dict += ", 'shape': (";
    auto cur = ds.begin();
    if (cur != ds.end()) {
      dict += std::to_string(*cur);
      dict += ',';
      ++cur;
      bool flag = false;
      while (cur != ds.end()) {
        if (flag) dict += ',';
        dict += std::to_string(*cur);
        ++cur;
        flag = true;
      }
    }
    dict += ") }";
    //pad with spaces so that header size is modulo 16 bytes. dict needs to end with \n
    dict.append(15 - (dict.size() + 10) % 16, ' ');
    dict += '\n';

    std::string header;
    header += static_cast<char>(0x93);
    header += "NUMPY";
    header += static_cast<char>(0x01); //major version of numpy format
    header += static_cast<char>(0x00); //minor version of numpy format
    header += toBytes(static_cast<uint16_t>(dict.size()), isLittleEndian());
    header += dict;

    return header;
  };

  struct NpyHeader {
    std::string dtype;
    bool fortran_order;
    std::vector<unsigned int> shape;
  };

  NpyHeader readNpyHeader(std::istream& is);

  template <typename T>
  void checkNpyHeader(std::istream& is, std::initializer_list<unsigned int> ds) {
    NpyHeader header = readNpyHeader(is);
    assert(header.dtype == numpy_dtype<T>());
    assert(header.shape.size() == ds.size());
    auto it = header.shape.cbegin();
    for (unsigned int d : ds) {
      assert(d == *it);
      ++it;
    }
  }
}

#endif //__MISC_H
