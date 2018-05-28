#include "misc.h"


using namespace std;
using namespace misc;

vector<string> misc::split(const string &s, char delim) {
  vector<string> ret;
  stringstream ss(s);

  string tmp;
  while (getline(ss, tmp, delim)) ret.push_back(move(tmp));
  return ret;
}

string misc::getField(const string &s, const string &desc, const string &lch, const string &rch) {
  auto tmp = s.find_first_not_of(lch, s.find(desc) + desc.size());
  return string(s, tmp, s.find_first_of(rch, tmp) - tmp);
}

bool misc::isLittleEndian() {
  union { unsigned char c[2] = {1, 0}; short s; } x;
  return x.s == 1;
}

NpyHeader misc::readNpyHeader(istream &is) {
  assert(is.get() == 0x93);
  assert(is.get() == 'N');
  assert(is.get() == 'U');
  assert(is.get() == 'M');
  assert(is.get() == 'P');
  assert(is.get() == 'Y');
  assert(is.get() == 0x01);
  assert(is.get() == 0x00);

  char* buf;
  buf = new char[2];
  is.read(buf, 2);
  auto dict_sz = fromBytes<uint16_t, const char*>(buf, buf + 2, isLittleEndian());
  delete[] buf;
  buf = new char[dict_sz];
  is.read(buf, dict_sz);
  string dict(buf, dict_sz);
  delete[] buf;
  assert(dict.back() == '\n');

  vector<unsigned int> shape;
  for (const string& x : split(getField(dict, "shape", "'\": (\t\n", ")"), ',')) {
    string tmp = getField(x, "", " \t\n", " \t\n");
    if (!tmp.empty()) shape.push_back(stoi(tmp));
  }

  return NpyHeader {
      getField(dict, "descr", "'\": \t\n", "'\""),
      getField(dict, "fortran_order", "'\": \t\n", ", \t\n").compare(0, 4, "True") == 0,
      shape
  };
}
