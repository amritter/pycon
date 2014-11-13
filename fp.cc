// This file is part of pycon: A framework for image reconstruction in X-ray
// Talbot-Lau interferometry.
//
// Copyright (C)
// 2012-2014 André Ritter (andre.ritter@fau.de)
#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"
#include <cmath>
#include <list>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>

namespace bp = boost::python;

inline npy_double
lc0(npy_double xi, npy_double c, npy_double sinv, npy_double x)
{
  return (x - xi * c) * sinv;
}

inline npy_double
lc1(npy_double xi, npy_double s, npy_double cinv, npy_double y)
{
  return (y + xi * s) * cinv;
}

inline long
get_index(npy_double start, npy_double pitch, npy_double c)
{
  return long((c - start) / pitch);
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

typedef std::list<Vertex> ray_type;

struct ray_to_tuple
{
  static PyObject*
  convert(const ray_type& ray)
  {
    bp::list indexes;
    bp::list weights;
    for (const auto& w : ray)
      {
        indexes.append(w.index);
        weights.append(w.weight);
      }
    return bp::incref(bp::make_tuple(indexes, weights).ptr());
  }
};

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
within_bounds(const npy_double c, const npy_double s, const npy_double xi,
    const npy_double start0, const npy_double end0, const npy_double start1,
    const npy_double end1)
{
  const npy_double cinv = 1. / c;
  const npy_double sinv = 1. / s;
  const npy_double p0start = (xi - start1 * s) * cinv;
  const npy_double p0end = (xi - end1 * s) * cinv;
  const npy_double p1start = (xi - start0 * s) * sinv;
  const npy_double p1end = (xi - end0 * s) * sinv;
  return (point_within_bounds(p0start, start0, end0)
      || point_within_bounds(p0end, start0, end0)
      || point_within_bounds(p1start, start1, end1)
      || point_within_bounds(p1end, start1, end1));

}

ray_type
siddon2d(npy_double theta, npy_double xi, size_t n0, size_t n1,
    npy_double width0, npy_double width1, npy_double center0,
    npy_double center1)
{
  ray_type ray;

  const npy_double s = std::sin(theta);
  const npy_double c = std::cos(theta);
  const npy_double pitch0 = width0 / n0;
  const npy_double pitch1 = width1 / n1;
  const npy_double start0 = -.5 * width0 + center0;
  const npy_double start1 = -.5 * width1 + center1;

  if (!within_bounds(c, s, xi, start0, start0 + width0, start1, start1+width1))
    {
      //std::cerr << "Not within" << xi*c << " " << xi*s << std::endl;
      return ray;
    }

  if (c == 0)
    {
      const long index1 = get_index(start1, pitch1, -xi * s);
      for (long index0 = 0; index0 < n0; ++index0)
        {
          ray.emplace_back(index0 + index1 * n0, pitch0);
        }
    }
  else if (s == 0)
    {
      const long index0 = get_index(start0, pitch0, xi * c);
      for (long index1 = 0; index1 < n1; ++index1)
        {
          ray.emplace_back(index0 + index1 * n0, pitch1);
        }
    }
  else
    {
      const npy_double sinv = 1.0 / s;
      const npy_double cinv = 1.0 / c;
      const npy_double lpitch0 = pitch0 * sinv;
      const npy_double lpitch1 = pitch1 * cinv;

      const npy_double lstart0 = lc0(xi, c, sinv, start0);
      const npy_double lstart1 = lc1(xi, s, cinv, start1);

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

template<size_t D, class T = npy_double, int typenum = NPY_DOUBLE>
  class ContiguousFixedDArray
  {
    PyObject* _pyo;
    PyArrayObject* _arr;
    size_t _shape[D];
    T* _data;
    size_t _size;
  public:
    ContiguousFixedDArray(bp::object o) :
        _pyo(o.ptr()), _arr(
            reinterpret_cast<PyArrayObject*>(PyArray_ContiguousFromAny(o.ptr(),
                typenum, D, D))), _shape(), _data(
            static_cast<T*>(PyArray_DATA(_arr))), _size(0)
    {
      std::copy_n(PyArray_SHAPE(_arr), D, _shape);
      _size = std::accumulate(_shape, _shape + D, 1, std::multiplies<size_t>());
    }
    ~ContiguousFixedDArray()
    {
      Py_DECREF(_arr);
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
    T
    operator[](size_t index) const
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
  };

typedef std::vector<ray_type> projector_type;

typedef ray_type
(*ray_func_ptr)(npy_double, npy_double, size_t, size_t, npy_double, npy_double,
    npy_double, npy_double);
template<ray_func_ptr ray_func>
  projector_type
  get_projector(bp::object thetas, bp::object xis, int n0, int n1,
      npy_double width0, npy_double width1, npy_double center0,
      npy_double center1)
  {
    ContiguousFixedDArray<1> arr_thetas(thetas);
    ContiguousFixedDArray<1> arr_xis(xis);
    projector_type projector;
    projector.reserve(arr_thetas.size() * arr_xis.size());
    for (auto theta : arr_thetas)
      {
        for (auto xi : arr_xis)
          {
            projector.push_back(
                ray_func(theta, xi, n0, n1, width0, width1, center0, center1));
          }
      }
    return projector;
  }

template<ray_func_ptr ray_func>
  ray_type
  get_differential_ray(npy_double theta, npy_double xi_low, npy_double xi_up,
      size_t n0, size_t n1, npy_double width0, npy_double width1,
      npy_double center0, npy_double center1)
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

template<ray_func_ptr ray_func>
  projector_type
  get_differential_projector(bp::object thetas, bp::object xis_low,
      bp::object xis_up, size_t n0, size_t n1, npy_double width0,
      npy_double width1, npy_double center0, npy_double center1)
  {
    ContiguousFixedDArray<1> arr_thetas(thetas);
    ContiguousFixedDArray<1> arr_xis_low(xis_low);
    ContiguousFixedDArray<1> arr_xis_up(xis_up);
    if (arr_xis_low.size() != arr_xis_up.size())
      {
        throw std::runtime_error(
            "Range of lower and upper xi values does not have equal size.");
      }
    projector_type projector;
    projector.reserve(arr_thetas.size() * arr_xis_low.size());
    for (auto theta : arr_thetas)
      {
        auto xi_up_it = arr_xis_up.begin();
        for (auto xi_low : arr_xis_low)
          {
            projector.push_back(
                get_differential_ray<ray_func>(theta, xi_low, *xi_up_it, n0, n1,
                    width0, width1, center0, center1));
            ++xi_up_it;
          }
      }
    return projector;
  }

projector_type
reverse_indexes(projector_type& projector, size_t n0, size_t n1)
{
  projector_type projector_voxelbased;
  projector_voxelbased.resize(n0 * n1);
  projector_type::size_type pixel_index = 0;
  for (const auto& ray : projector)
    {
      for (const auto& vertex : ray)
        {
          projector_voxelbased[vertex.index].emplace_back(pixel_index,
              vertex.weight);
        }
      ++pixel_index;
    }
  for (auto& ray : projector_voxelbased)
    {
      ray.sort();
    }
  return projector_voxelbased;
}

bp::object
project(const projector_type& projector, bp::object volume)
{
  bp::list projection;
  ContiguousFixedDArray<2> arr_volume(volume);
  for (const auto& ray : projector)
    {
      npy_double value = 0.;
      for (const auto& vertex : ray)
        {
          value += arr_volume[vertex.index] * vertex.weight;
        }
      projection.append(value);
    }
  return projection;
}

BOOST_PYTHON_MODULE(fp)
{
  import_array();
  bp::to_python_converter<ray_type, ray_to_tuple>();

  bp::class_<projector_type>("projector_type", bp::no_init).def("__iter__",
      bp::iterator<projector_type>()).def("__len__", &projector_type::size);

  bp::def("get_index", get_index, (bp::arg("start"), "pitch", "c"));
  bp::def("siddon2d", siddon2d,
      (bp::arg("theta"), "xi", "n0", "n1", "width0", "width1", bp::arg(
          "center0") = 0., bp::arg("center1") = 0.));
  bp::def("siddon2d_differential", get_differential_ray<siddon2d>,
      (bp::arg("theta"), "xi_low", "xi_up", "n0", "n1", "width0", "width1", bp::arg(
          "center0") = 0., bp::arg("center1") = 0.));
  bp::def("projector_siddon2d", get_projector<siddon2d>,
      (bp::arg("thetas"), "xis", "n0", "n1", "width0", "width1", bp::arg(
          "center0") = 0., bp::arg("center1") = 0.));
  bp::def("differential_projector_siddon2d",
      get_differential_projector<siddon2d>,
      (bp::arg("thetas"), "xis_low", "xis_up", "n0", "n1", "width0", "width1", bp::arg(
          "center0") = 0., bp::arg("center1") = 0.));
  bp::def("reverse_indexes", reverse_indexes,
      (bp::arg("projector"), "n0", "n1"));
  bp::def("project", project, (bp::arg("projector"), "volume"));
}
