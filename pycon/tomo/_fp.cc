// This file is part of pycon: A framework for image reconstruction in X-ray
// Talbot-Lau interferometry.
//
// Copyright (C)
// 2012-2014 André Ritter (andre.ritter@fau.de)
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"
#include <iostream>
#include <cmath>
#include <list>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>

struct Vertex;
typedef std::list<Vertex> ray_type;
typedef std::vector<ray_type> projector_type;
typedef ray_type
(*ray_func_ptr)(npy_double, npy_double, size_t, size_t, npy_double, npy_double,
    npy_double, npy_double);

inline npy_double
x_by_lambda(npy_double xi, npy_double lambda, npy_double c, npy_double s)
{
  return (lambda * s + xi * c);
}

inline npy_double
y_by_lambda(npy_double xi, npy_double lambda, npy_double c, npy_double s)
{
  return (lambda * c - xi * s);
}

inline npy_double
xi_by_xy(npy_double x, npy_double y, npy_double c, npy_double s)
{
  return (x * c - y * s);
}

inline npy_double
x_by_y(npy_double xi, npy_double s, npy_double cinv, npy_double y)
{
  return (y * s + xi) * cinv;
}

inline npy_double
y_by_x(npy_double xi, npy_double c, npy_double sinv, npy_double x)
{
  return (x * c - xi) * sinv;
}

inline npy_double
lambda_by_x(npy_double xi, npy_double c, npy_double sinv, npy_double x)
{
  return (x - xi * c) * sinv;
}

inline npy_double
lambda_by_y(npy_double xi, npy_double s, npy_double cinv, npy_double y)
{
  return (y + xi * s) * cinv;
}

inline long
get_index(npy_double start, npy_double pitch, npy_double pos)
{
  return long((pos - start) / pitch);
}

struct Vertex
{
  typedef size_t index_type;
  typedef npy_double weight_type;
  index_type index;
  weight_type weight;
  Vertex(index_type _index, weight_type weight) :
      index(_index), weight(weight)
  {
  }
};

bool
operator<(const Vertex& lop, const Vertex& rop)
{
  return (lop.weight < rop.weight);
}

template<size_t D, class T = npy_double, int typenum = NPY_DOUBLE>
  class ContiguousFixedDArray
  {
    PyObject* _pyo;
    size_t _shape[D];
    T* _data;
    size_t _size;
  public:
    ContiguousFixedDArray(PyObject* object) :
        _pyo(PyArray_ContiguousFromAny(object, typenum, D, D)), _shape(), _data(), _size()
    {
      auto arr = reinterpret_cast<PyArrayObject*>(_pyo);
      std::copy_n(PyArray_SHAPE(arr), D, _shape);
      _data = static_cast<T*>(PyArray_DATA(arr));
      _size = std::accumulate(_shape, _shape + D, 1, std::multiplies<size_t>());
    }
    ContiguousFixedDArray(npy_intp* shape) :
        ContiguousFixedDArray(PyArray_ZEROS(D, shape, typenum, 0))
    {
    }
    ~ContiguousFixedDArray()
    {
      Py_DECREF(_pyo);
    }
    template<size_t dim>
      size_t
      shape() const
      {
        if (dim < 0 || dim >= D)
          {
            throw(std::runtime_error("Dimension out of range."));
          }
        return _shape[dim];
      }
    size_t
    ndim() const
    {
      return D;
    }
    size_t
    size() const
    {
      return _size;
    }
    const T&
    operator[](size_t index) const
    {
      return _data[index];
    }
    T&
    operator[](size_t index)
    {
      return _data[index];
    }
    const T*
    begin() const
    {
      return _data;
    }
    T*
    begin()
    {
      return _data;
    }
    const T*
    end() const
    {
      return _data + size();
    }
    T*
    end()
    {
      return _data + size();
    }
    PyObject*
    ptr() const
    {
      return _pyo;
    }
  };

PyObject*
ray_to_PyObject(const ray_type& ray)
{
  npy_intp size[1] =
    { (npy_intp) ray.size() };
  ContiguousFixedDArray<1, npy_intp, NPY_INTP> indexes(size);
  ContiguousFixedDArray<1> weights(size);
  size_t i = 0;
  for (const auto& w : ray)
    {
      indexes[i] = w.index;
      weights[i] = w.weight;
      ++i;
    }
  return Py_BuildValue("OO", indexes.ptr(), weights.ptr());
}

ray_type
siddon_vertices(npy_double lstart, npy_double lpitch, size_t size)
{
  ray_type ray;
  npy_double length = lstart;
  if (lpitch >= 0)
    {
      for (long index = 0; index <= size; ++index)
        {
          ray.push_back(Vertex(index, length));
          length += lpitch;
        }
    }
  else
    {
      for (long index = 0; index <= size; ++index)
        {
          ray.push_front(Vertex(index - 1, length));
          length += lpitch;
        }
    }
  return ray;
}

bool
point_within_bounds(const npy_double c, const npy_double start,
    const npy_double end)
{
  return (start <= c && c < end);
}

bool
within_bounds(const npy_double c, const npy_double s, const npy_double cinv,
    const npy_double sinv, const npy_double xi, const npy_double startx,
    const npy_double endx, const npy_double starty, const npy_double endy)
{
  const npy_double x_starty = x_by_y(xi, s, cinv, starty);
  const npy_double x_endy = x_by_y(xi, s, cinv, endy);
  const npy_double y_startx = y_by_x(xi, c, sinv, startx);
  const npy_double y_endx = y_by_x(xi, c, sinv, endx);
  return (point_within_bounds(x_starty, startx, endx)
      || point_within_bounds(x_endy, startx, endx)
      || point_within_bounds(y_startx, starty, endy)
      || point_within_bounds(y_endx, starty, endy));

}

ray_type
siddon2d(npy_double theta, npy_double xi, size_t n0, size_t n1,
    npy_double width0, npy_double width1, npy_double center0,
    npy_double center1)
{
  ray_type ray;

  const npy_double s = std::sin(theta);
  const npy_double c = std::cos(theta);
  const npy_double cinv = 1. / c;
  const npy_double sinv = 1. / s;
  const npy_double pitch0 = width0 / n0;
  const npy_double pitch1 = width1 / n1;
  const npy_double start0 = -.5 * width0 + center0;
  const npy_double start1 = -.5 * width1 + center1;

  if (!within_bounds(c, s, cinv, sinv, xi, start0, start0 + width0, start1,
      start1 + width1))
    {
      return ray;
    }

  if (c == 0) // cos(theta) == 0 => Ray parallel to x.
    {
      const long index1 = get_index(start1, pitch1, -xi * s);
      for (long index0 = 0; index0 < n0; ++index0)
        {
          ray.emplace_back(index0 + index1 * n0, pitch0);
        }
    }
  else if (s == 0) // sin(theta) == 0 => Ray parallel to y.
    {
      const long index0 = get_index(start0, pitch0, xi * c);
      for (long index1 = 0; index1 < n1; ++index1)
        {
          ray.emplace_back(index0 + index1 * n0, pitch1);
        }
    }
  else
    {
      const npy_double lpitch0 = lambda_by_x(0., c, sinv, pitch0);
      const npy_double lpitch1 = lambda_by_y(0., s, cinv, pitch1);

      const npy_double lstart0 = lambda_by_x(xi, c, sinv, start0);
      const npy_double lstart1 = lambda_by_y(xi, s, cinv, start1);

      const ray_type vertices0 = siddon_vertices(lstart0, lpitch0, n0);
      const ray_type vertices1 = siddon_vertices(lstart1, lpitch1, n1);

      auto it0 = vertices0.begin();
      const auto it0_end = vertices0.end();
      auto it1 = vertices1.begin();
      const auto it1_end = vertices1.end();

      auto index0_last = it0->index;
      auto index1_last = it1->index;

      if (*it0 < *it1)
        {
          while (*it0 < *it1 && it0 != it0_end)
            {
              index0_last = it0->index;
              ++it0;
            }
        }
      else
        {
          while (*it1 < *it0 && it1 != it1_end)
            {
              index1_last = it1->index;
              ++it1;
            }
        }

      npy_double weight_limit = 1e-12
          * std::sqrt(pitch0 * pitch0 + pitch1 * pitch1);
      while (it0 != it0_end && it1 != it1_end)
        {
          npy_double weight = 0.;
          if (*it0 < *it1)
            {
              auto it0_next = it0;
              ++it0_next;
              if (it0_next != it0_end)
                {
                  index0_last = it0->index;
                  if (*it0_next < *it1)
                    {
                      weight = it0_next->weight - it0->weight;
                    }
                  else
                    {
                      weight = it1->weight - it0->weight;
                    }
                }
              it0 = it0_next;
            }
          else
            {
              auto it1_next = it1;
              ++it1_next;
              if (it1_next != it1_end)
                {
                  index1_last = it1->index;
                  if (*it1_next < *it0)
                    {
                      weight = it1_next->weight - it1->weight;
                    }
                  else
                    {
                      weight = it0->weight - it1->weight;
                    }
                }
              it1 = it1_next;
            }

          if (weight > weight_limit)
            {
              ray.emplace_back(index0_last + index1_last * n0, weight);
            }
        }
    }
  return ray;
}

template<ray_func_ptr ray_func>
  ray_type
  ray_diff(npy_double theta, npy_double xi_low, npy_double xi_up, size_t n0,
      size_t n1, npy_double width0, npy_double width1, npy_double center0,
      npy_double center1)
  {
    auto ray_low = ray_func(theta, xi_low, n0, n1, width0, width1, center0,
        center1);
    auto ray_up = ray_func(theta, xi_up, n0, n1, width0, width1, center0,
        center1);
    const npy_double pitch0 = width0 / n0;
    const npy_double pitch1 = width1 / n1;
    const npy_double weight_limit = 1e-12
        * std::sqrt(pitch0 * pitch0 + pitch1 * pitch1);
    for (auto& v_up : ray_up)
      {
        auto it = find_if(ray_low.begin(), ray_low.end(),
            [v_up](const Vertex& v)
              { return v.index==v_up.index;});
        if (it != ray_low.end())
          {
            v_up.weight -= it->weight;
            ray_low.erase(it);
          }
      }
    for (auto& v_low : ray_low)
      {
        v_low.weight = -v_low.weight;
      }
    ray_up.splice(ray_up.end(), ray_low);
    const auto ray_up_end = ray_up.end();
    for (auto v_it = ray_up.begin(); v_it != ray_up_end; ++v_it)
      {
        if (std::abs(v_it->weight) <= weight_limit)
          {
            ray_up.erase(v_it);
          }
      }
    ray_up.sort();
    return ray_up;
  }

extern "C"
{

  static PyObject*
  _fp_ray(PyObject* self, PyObject* args, PyObject* kwargs)
  {
    npy_double theta;
    npy_double xi;
    npy_int nx;
    npy_int ny;
    npy_double widthx;
    npy_double widthy;
    npy_double centerx = 0.;
    npy_double centery = 0.;
    const char* method = "siddon";

    static const char* kwlist[] =
      { "theta", "xi", "nx", "ny", "widthx", "widthy", "centerx", "centery",
          "method", nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddiidd|dds:ray",
        const_cast<char**>(kwlist), &theta, &xi, &nx, &ny, &widthx, &widthy,
        &centery, &centery, &method))
      {
        return nullptr;
      }
    return ray_to_PyObject(
        siddon2d(theta, xi, nx, ny, widthx, widthy, centerx, centery));
  }

  static PyObject*
  _fp_ray_diff(PyObject* self, PyObject* args, PyObject* kwargs)
  {
    npy_double theta;
    npy_double xi_lo;
    npy_double xi_hi;
    npy_int nx;
    npy_int ny;
    npy_double widthx;
    npy_double widthy;
    npy_double centerx = 0.;
    npy_double centery = 0.;
    const char* method = "siddon";

    static const char* kwlist[] =
      { "theta", "xi_lo", "xi_hi", "nx", "ny", "widthx", "widthy", "centerx",
          "centery", "method", nullptr };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dddiidd|dds:ray_diff",
        const_cast<char**>(kwlist), &theta, &xi_lo, &xi_hi, &nx, &ny, &widthx,
        &widthy, &centery, &centery, &method))
      {
        return nullptr;
      }
    return ray_to_PyObject(
        ray_diff<siddon2d>(theta, xi_lo, xi_hi, nx, ny, widthx, widthy, centerx,
            centery));
  }

  static PyMethodDef _fp_methods[] =
    {
          { "ray", (PyCFunction) _fp_ray, METH_VARARGS | METH_KEYWORDS,
              "Get ray." },
          { "ray_diff", (PyCFunction) _fp_ray_diff, METH_VARARGS
              | METH_KEYWORDS, "Get differential ray." },
          { nullptr, nullptr, 0, nullptr } /* Sentinel */
    };

  PyMODINIT_FUNC
  init_fp(void)
  {
    import_array();

    (void) Py_InitModule("_fp", _fp_methods);
  }

}
