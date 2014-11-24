// This file is part of pycon: A framework for image reconstruction in X-ray
// Talbot-Lau interferometry.
//
// Copyright (C)
// 2012-2014 André Ritter (andre.ritter@fau.de)
#ifndef _fp_h
#define _fp_h
#include <list>
#include <vector>

inline double
x_by_lambda(double xi, double lambda, double c, double s)
{
  return (lambda * s + xi * c);
}

inline double
y_by_lambda(double xi, double lambda, double c, double s)
{
  return (lambda * c - xi * s);
}

inline double
xi_by_xy(double x, double y, double c, double s)
{
  return (x * c - y * s);
}

inline double
x_by_y(double xi, double s, double cinv, double y)
{
  return (y * s + xi) * cinv;
}

inline double
y_by_x(double xi, double c, double sinv, double x)
{
  return (x * c - xi) * sinv;
}

inline double
lambda_by_x(double xi, double c, double sinv, double x)
{
  return (x - xi * c) * sinv;
}

inline double
lambda_by_y(double xi, double s, double cinv, double y)
{
  return (y + xi * s) * cinv;
}

inline long
get_index(double start, double pitch, double pos)
{
  return long((pos - start) / pitch);
}

struct Vertex
{
  typedef size_t index_type;
  typedef double weight_type;
  index_type index;
  weight_type weight;
  Vertex(index_type index, weight_type weight);
};

bool
operator<(const Vertex& lop, const Vertex& rop);

typedef std::list<Vertex> ray_type;
typedef std::vector<ray_type> projector_type;
typedef std::vector<size_t> shape_type;
typedef ray_type
(*ray_func_ptr)(double, double, size_t, size_t, double, double, double, double);

ray_type
siddon2d(double theta, double xi, size_t n0, size_t n1, double width0,
    double width1, double center0, double center1);

bool
point_within_bounds(double pos, double start, double end);

bool
ray_within_bounds(double c, double s, double cinv, double sinv, double xi,
    double startx, double endx, double starty, double endy);

#endif // _fp_h
