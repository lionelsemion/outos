from __future__ import annotations

from math import atan2, cos, inf, sin, sqrt, floor
import cv2
from numpy import sign

BIG = 1e10


# def intersect(pt1: Vec, pt2: Vec, pt3: Vec, pt4: Vec):
#     # https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect

#     X1, Y1 = pt1.xy
#     X2, Y2 = pt2.xy
#     X3, Y3 = pt3.xy
#     X4, Y4 = pt4.xy

#     Ia = [max(min(X1, X2), min(X3, X4)), min(max(X1, X2), max(X3, X4))]
#     if max(X1, X2) < min(X3, X4):
#         return False  # There is no mutual abcisses

#     A1 = (Y1 - Y2) / (X1 - X2) if X1 != X2 else BIG
#     A2 = (Y3 - Y4) / (X3 - X4) if X3 != X4 else BIG
#     b1 = Y2 - A1 * X2
#     b2 = Y4 - A2 * X4

#     if A1 - A2 == 0:
#         return False  # Parallel segments

#     Xa = (b2 - b1) / (A1 - A2)

#     if (Xa < max(min(X1, X2), min(X3, X4))) or (Xa > min(max(X1, X2), max(X3, X4))):
#         return False  # intersection is out of bound
#     return True


def intersect(a, b, c, d):
    """Verifies if closed segments a, b, c, d do intersect."""

    def side(a, b, c):
        """Returns a position of the point c relative to the line going through a and b
        Points a, b are expected to be different"""
        d = (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)
        return 1 if d > 0 else (-1 if d < 0 else 0)

    def is_point_in_closed_segment(a, b, c):
        """Returns True if c is inside closed segment, False otherwise.
        a, b, c are expected to be collinear"""
        if a.x < b.x:
            return a.x <= c.x and c.x <= b.x
        if b.x < a.x:
            return b.x <= c.x and c.x <= a.x

        if a.y < b.y:
            return a.y <= c.y and c.y <= b.y
        if b.y < a.y:
            return b.y <= c.y and c.y <= a.y

        return a.x == c.x and a.y == c.y

    if a == b:
        return a == c or a == d
    if c == d:
        return c == a or c == b

    s1 = side(a, b, c)
    s2 = side(a, b, d)

    # All points are collinear
    if s1 == 0 and s2 == 0:
        return (
            is_point_in_closed_segment(a, b, c)
            or is_point_in_closed_segment(a, b, d)
            or is_point_in_closed_segment(c, d, a)
            or is_point_in_closed_segment(c, d, b)
        )

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    s1 = side(c, d, a)
    s2 = side(c, d, b)

    # No touching and on the same side
    if s1 and s1 == s2:
        return False

    return True


class Vec:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def rho(self):
        return sqrt(self.x * self.x + self.y * self.y)

    @property
    def rho2(self):
        return self.x * self.x + self.y * self.y

    @rho.setter
    def rho(self, value):
        scale = value / self.rho
        self.x *= scale
        self.y *= scale

    @property
    def phi(self):
        return atan2(self.y, self.x)

    @phi.setter
    def phi(self, value):
        self.x, self.y = self.rho * cos(value), self.rho * sin(value)

    def __add__(self, other):
        return Vec(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return Vec(-self.x, -self.y)

    def __sub__(self, other):
        return self + -other

    def __repr__(self) -> str:
        return f"{self.x} {self.y}"

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def copy(self):
        return Vec(self.x, self.y)

    def __round__(self):
        return Vec(round(self.x), round(self.y))

    def __floor__(self):
        return Vec(round(self.x), round(self.y))

    def __mul__(self, factor):
        return Vec(self.x * factor, self.y * factor)

    __rmul__ = __mul__


class Line:
    vec1: Vec
    vec2: Vec

    def __init__(self, vec1: Vec, vec2: Vec) -> None:
        self.vec1 = vec1
        self.vec2 = vec2

    def intersects(self, other: Line) -> bool:
        return intersect(self.vec1, self.vec2, other.vec1, other.vec2)

    def draw(self, img, color=(255, 255, 255), scale=1, width=1):
        return cv2.line(
            img,
            round(scale * self.vec1).xy,
            round(scale * self.vec2).xy,
            color,
            width,
            cv2.LINE_AA,
        )
