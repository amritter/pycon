// This file is part of pycon: A framework for image reconstruction in X-ray
// Talbot-Lau interferometry.
//
// Copyright (C)
// 2012-2014 André Ritter (andre.ritter@fau.de)
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include "numpy/arrayobject.h"
#include "fp.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>

Vertex::Vertex(index_type index, weight_type weight) :
    index(index), weight(weight)
{
}

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

bool
point_within_bounds(double pos, double start, double end)
{
  return (start <= pos && pos < end);
}

bool
ray_within_bounds(double c, double s, double cinv, double sinv, double xi,
    double startx, double endx, double starty, double endy)
{
  const double x_starty = x_by_y(xi, s, cinv, starty);
  const double x_endy = x_by_y(xi, s, cinv, endy);
  const double y_startx = y_by_x(xi, c, sinv, startx);
  const double y_endx = y_by_x(xi, c, sinv, endx);
  return (point_within_bounds(x_starty, startx, endx)
      || point_within_bounds(x_endy, startx, endx)
      || point_within_bounds(y_startx, starty, endy)
      || point_within_bounds(y_endx, starty, endy));

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

template<ray_func_ptr ray_func, class Iterator>
  projector_type
  get_projector(Iterator thetas_begin, const Iterator thetas_end,
      Iterator xis_begin, const Iterator xis_end, size_t n, int n0, int n1,
      npy_double width0, npy_double width1, npy_double center0,
      npy_double center1)
  {
    projector_type projector;
    projector.reserve(n);
    for (auto thetas_it = thetas_begin; thetas_it != thetas_end; ++thetas_it)
      {
        for (auto xis_it = xis_begin; xis_it != xis_end; ++xis_it)
          {
            projector.push_back(
                ray_func(*thetas_it, *xis_it, n0, n1, width0, width1, center0,
                    center1));
          }
      }
    return projector;
  }

template<ray_func_ptr ray_func, class Iterator>
  projector_type
  get_differential_projector(Iterator thetas_begin, const Iterator thetas_end,
      Iterator xis_begin, const Iterator xis_end, Iterator xis_diff_begin,
      size_t n, size_t n0, size_t n1, npy_double width0, npy_double width1,
      npy_double center0, npy_double center1)
  {
    projector_type projector;
    projector.reserve(n);
    for (auto thetas_it = thetas_begin; thetas_it != thetas_end; ++thetas_it)
      {
        auto xis_diff_it = xis_diff_begin;
        for (auto xis_it = xis_begin; xis_it != xis_end; ++xis_it)
          {
            projector.push_back(
                ray_diff<ray_func>(*thetas_it, *xis_it - .5 * (*xis_diff_it),
                    *xis_it + .5 * (*xis_diff_it), n0, n1, width0, width1,
                    center0, center1));
            ++xis_diff_it;
          }
      }
    return projector;
  }

extern "C"
{
///////////////////////////////
// Definition of _fp.Projector.
  typedef struct
  {
    PyObject_HEAD
    shape_type* _shape_i;
    shape_type* _shape_o;
    projector_type* _projector;
  } _fp_Projector;

  static void
  _fp_Projector_dealloc(_fp_Projector* self)
  {
    if (self->_shape_i != nullptr)
      {
        delete self->_shape_i;
      }
    if (self->_shape_o != nullptr)
      {
        delete self->_shape_o;
      }
    if (self->_projector != nullptr)
      {
        delete self->_projector;
      }
    self->ob_type->tp_free(reinterpret_cast<PyObject*>(self));
  }

  static PyObject*
  _fp_Projector_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
  {
    _fp_Projector *self;
    self = reinterpret_cast<_fp_Projector*>(type->tp_alloc(type, 0));

    if (self != nullptr)
      {
        self->_shape_i = new shape_type();
        self->_shape_o = new shape_type();
        self->_projector = new projector_type();
      }

    return reinterpret_cast<PyObject*>(self);
  }

  static int
  _fp_Projector_init(_fp_Projector *self, PyObject *args, PyObject *kwds)
  {
    return 0;
  }

  PyObject*
  _fp_Projector_transposed(_fp_Projector* self)
  {
    const size_t size_o =
        (self->_shape_o->size() == 0) ?
            0 :
            std::accumulate(self->_shape_o->begin(), self->_shape_o->end(), 1,
                std::multiplies<size_t>());
    _fp_Projector* ret = reinterpret_cast<_fp_Projector*>(_fp_Projector_new(
        self->ob_type, nullptr, nullptr));

    ret->_projector->resize(size_o);
    *(ret->_shape_i) = *(self->_shape_o);
    *(ret->_shape_o) = *(self->_shape_i);

    size_t pixel_index = 0;
    for (const auto& ray : *(self->_projector))
      {
        for (const auto& vertex : ray)
          {
            (*(ret->_projector))[vertex.index].emplace_back(pixel_index,
                vertex.weight);
          }
        ++pixel_index;
      }
    for (auto& ray : *(ret->_projector))
      {
        ray.sort();
      }
    return reinterpret_cast<PyObject*>(ret);
  }

  PyObject*
  _fp_Projector_project(const _fp_Projector* self, PyObject *args)
  {
    PyObject* volume = nullptr;

    if (!PyArg_ParseTuple(args, "O", &volume))
      {
        return nullptr;
      }

    ContiguousFixedDArray<2> arr_volume(volume);
    npy_intp* shape_o = new npy_intp[self->_shape_o->size()];
    std::copy(self->_shape_o->begin(), self->_shape_o->end(), shape_o);
    ContiguousFixedDArray<2> arr_projection(shape_o);
    delete shape_o;

    if (arr_volume.ptr() == nullptr || arr_projection.ptr() == nullptr)
      {
        return nullptr;
      }

    size_t index = 0;
    for (const auto& ray : *(self->_projector))
      {
        npy_double value = 0.;
        for (const auto& vertex : ray)
          {
            value += arr_volume[vertex.index] * vertex.weight;
          }
        arr_projection[index] = value;
        ++index;
      }
    return reinterpret_cast<PyObject*>(arr_projection.ptr());
  }

  static PyMethodDef _fp_Projector_methods[] =
    {
          { "transposed", (PyCFunction) _fp_Projector_transposed, METH_NOARGS,
              "." },
          { "project", (PyCFunction) _fp_Projector_project, METH_VARARGS, "." },
          { nullptr } /* Sentinel */
    };

  static Py_ssize_t
  _fp_Projector_sq_length(PyObject *o)
  {
    return reinterpret_cast<_fp_Projector*>(o)->_projector->size();
  }

  static PyObject*
  _fp_Projector_sq_item(PyObject *o, Py_ssize_t index)
  {
    auto self = reinterpret_cast<_fp_Projector*>(o);
    if (index < 0 && index >= self->_projector->size())
      {
        PyErr_SetString(PyExc_ValueError, "Index is out of bounds.");
        return nullptr;
      }
    return ray_to_PyObject((*(self->_projector))[index]);
  }

  static PySequenceMethods _fp_ProjectorSequenceMethods =
    { _fp_Projector_sq_length, // sq_length
        0, // sq_concat
        0, // sq_repeat
        _fp_Projector_sq_item, // sq_item
        0, 0, 0, 0 };

static PyTypeObject _fp_ProjectorType =
  {
    PyObject_HEAD_INIT(nullptr)
    0, /*ob_size*/
    "_fp.Projector", /*tp_name*/
    sizeof(_fp_Projector), /*tp_basicsize*/
    0, /*tp_itemsize*/
    (destructor)_fp_Projector_dealloc, /*tp_dealloc*/
    0, /*tp_print*/
    0, /*tp_getattr*/
    0, /*tp_setattr*/
    0, /*tp_compare*/
    0, /*tp_repr*/
    0, /*tp_as_number*/
    &_fp_ProjectorSequenceMethods, /*tp_as_sequence*/
    0, /*tp_as_mapping*/
    0, /*tp_hash */
    0, /*tp_call*/
    0, /*tp_str*/
    0, /*tp_getattro*/
    0, /*tp_setattro*/
    0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "A sparse matrix storing the projection coefficients.", /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    _fp_Projector_methods, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0, /* tp_base */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)_fp_Projector_init, /* tp_init */
    0, /* tp_alloc */
    _fp_Projector_new, /* tp_new */
  }
;

///////////////
// _fp methods.

static PyObject*
_fp_siddon2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  double theta;
  double xi;
  int nx;
  int ny;
  double widthx;
  double widthy;
  double centerx = 0.;
  double centery = 0.;
  double xi_diff = NAN;

  static const char* kwlist[] =
    { "theta", "xi", "nx", "ny", "widthx", "widthy", "centerx", "centery",
        "xi_diff", nullptr };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddiidd|ddd:ray",
      const_cast<char**>(kwlist), &theta, &xi, &nx, &ny, &widthx, &widthy,
      &centery, &centery, &xi_diff))
    {
      return nullptr;
    }
  if (xi_diff == xi_diff) // xi_diff is set.
    {
      return ray_to_PyObject(
          ray_diff<siddon2d>(theta, xi - .5 * xi_diff, xi + .5 * xi_diff, nx,
              ny, widthx, widthy, centerx, centery));
    }
  return ray_to_PyObject(
      siddon2d(theta, xi, nx, ny, widthx, widthy, centerx, centery));
}

static PyObject*
_fp_projector_siddon2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  PyObject* thetas = nullptr;
  PyObject* xis = nullptr;
  PyObject* xis_diff = nullptr;
  int nx;
  int ny;
  double widthx;
  double widthy;
  double centerx = 0.;
  double centery = 0.;

  static const char* kwlist[] =
    { "thetas", "xis", "nx", "ny", "widthx", "widthy", "centerx", "centery",
        "xis_diff", nullptr };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOiidd|ddO:projector_siddon",
      const_cast<char**>(kwlist), &thetas, &xis, &nx, &ny, &widthx, &widthy,
      &centery, &centery, &xis_diff))
    {
      return nullptr;
    }

  ContiguousFixedDArray<1> arr_thetas(thetas);
  ContiguousFixedDArray<1> arr_xis(xis);

  _fp_Projector* ret = reinterpret_cast<_fp_Projector*>(PyObject_CallObject(
      reinterpret_cast<PyObject*>(&_fp_ProjectorType), nullptr));

  ret->_shape_o->push_back(arr_thetas.size());
  ret->_shape_o->push_back(arr_xis.size());
  ret->_shape_i->push_back(ny);
  ret->_shape_i->push_back(nx);

  size_t n = arr_thetas.size() * arr_xis.size();

  if (xis_diff != nullptr)
    {
      ContiguousFixedDArray<1> arr_xis_diff(xis_diff);
      if (arr_xis.size() != arr_xis_diff.size())
        {
          Py_DECREF(ret);
          PyErr_SetString(PyExc_ValueError,
              "Len of ndarray xis_diff does not match xis.");
          return nullptr;
        }
      *(ret->_projector) = get_differential_projector<siddon2d>(
          arr_thetas.begin(), arr_thetas.end(), arr_xis.begin(), arr_xis.end(),
          arr_xis_diff.begin(), n, nx, ny, widthx, widthy, centerx, centery);
    }
  else
    {
      *(ret->_projector) = get_projector<siddon2d>(arr_thetas.begin(),
          arr_thetas.end(), arr_xis.begin(), arr_xis.end(), n, nx, ny, widthx,
          widthy, centerx, centery);
    }
  return reinterpret_cast<PyObject*>(ret);
}

static PyMethodDef _fp_methods[] =
  {
    { "siddon2d", (PyCFunction) _fp_siddon2d, METH_VARARGS | METH_KEYWORDS,
        "Get siddon ray coefficients." },
    { "projector_siddon2d", (PyCFunction) _fp_projector_siddon2d, METH_VARARGS
        | METH_KEYWORDS, "Get siddon projector." },
    { nullptr, nullptr, 0, nullptr } /* Sentinel */
  };

PyMODINIT_FUNC
init_fp(void)
{
  import_array();

  if (PyType_Ready(&_fp_ProjectorType) < 0)
    return;

  auto* module = Py_InitModule("_fp", _fp_methods);

  Py_INCREF (&_fp_ProjectorType);
  PyModule_AddObject(module, "Projector",
      reinterpret_cast<PyObject *>(&_fp_ProjectorType));
}

}
