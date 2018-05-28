#include "ReaderLines.h"

#include <utility>

using namespace std;

ReaderLines::ReaderLines(const string &file_name) {
  file.open(file_name);
  if (!getline(file, cache)) {
    empty_flag = true;
    file.close();
  }
}

string ReaderLines::next() {
  string ret = move(cache);
  if (!getline(file, cache)) {
    empty_flag = true;
    file.close();
  }
  if (ret.back() == '\r') ret.pop_back();
  return ret;
}
