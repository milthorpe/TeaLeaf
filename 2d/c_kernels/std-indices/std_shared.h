#pragma once

#include <cassert>
#include <ostream>

struct Range2d {
  const int fromX, toX;
  const int fromY, toY;

  constexpr Range2d(int fromX, int fromY, int toX, int toY) : fromX(fromX), toX(toX), fromY(fromY), toY(toY) {
    assert(fromX < toX);
    assert(fromY < toY);
    assert(sizeX() != 0);
    assert(sizeY() != 0);
  }
  [[nodiscard]] constexpr int sizeX() const { return toX - fromX; }
  [[nodiscard]] constexpr int sizeY() const { return toY - fromY; }
  [[nodiscard]] constexpr int sizeXY() const { return sizeX() * sizeY(); }
  friend std::ostream &operator<<(std::ostream &os, const Range2d &d) {
    os << "Range2d{"
       << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX() << ")]"
       << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY() << ")]"
       << "}";
    return os;
  }
};