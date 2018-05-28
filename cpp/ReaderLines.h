#ifndef __READERLINES_H
#define __READERLINES_H

#include <fstream>
#include <string>

#include "Iterator.h"

class ReaderLines : public Iterator<std::string> {

  std::ifstream file;
  std::string cache;
  bool empty_flag = false;

public:
  explicit ReaderLines(const std::string& file_name);
  bool empty() const override { return empty_flag; }
  std::string next() override;
};

#endif //__READERLINES_H
