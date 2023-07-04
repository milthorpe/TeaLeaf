#pragma once

#include <cassert>
#include <ostream>

template <typename N = int> struct Range2d {
  const N fromX, toX;
  const N fromY, toY;

  constexpr Range2d(N fromX, N fromY, N toX, N toY) : fromX(fromX), toX(toX), fromY(fromY), toY(toY) {
    assert(fromX < toX);
    assert(fromY < toY);
    assert(sizeX() != 0);
    assert(sizeY() != 0);
  }
  [[nodiscard]] constexpr N sizeX() const { return toX - fromX; }
  [[nodiscard]] constexpr N sizeY() const { return toY - fromY; }
  [[nodiscard]] constexpr N sizeXY() const { return sizeX() * sizeY(); }
  friend std::ostream &operator<<(std::ostream &os, const Range2d &d) {
    os << "Range2d{"
       << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX() << ")]"
       << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY() << ")]"
       << "}";
    return os;
  }
};