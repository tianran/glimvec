#ifndef __ITERATOR_H
#define __ITERATOR_H

template <typename T>
class Iterator {

public:
  virtual bool empty() const = 0;
  virtual T next() = 0;
};

#endif //__ITERATOR_H
