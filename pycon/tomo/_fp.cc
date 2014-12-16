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
cmp_less_index(const Vertex& a, const Vertex& b)
{
  return a.index < b.index;
}

bool
cmp_less_weight(const Vertex& a, const Vertex& b)
{
  return a.weight < b.weight;
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
        _pyo(PyArray_ZEROS(D, shape, typenum, 0)), _shape(), _data(
            static_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(_pyo)))), _size(
            std::accumulate(shape, shape + D, 1, std::multiplies<size_t>()))
    {
      std::copy_n(shape, D, _shape);
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
    PyObject*
    new_ref() const
    {
      Py_INCREF(ptr());
      return ptr();
    }
  };

PyObject*
ray_to_PyObject(const ray_type& ray)
{
  PyObject* indexes = PyList_New(ray.size());
  PyObject* weights = PyList_New(ray.size());
  if (indexes == nullptr || weights == nullptr)
    {
      Py_XDECREF(indexes);
      Py_XDECREF(weights);
      return nullptr;
    }
  Py_ssize_t index = 0;
  for (const auto& v : ray)
    {
      PyList_SetItem(indexes, index, PyInt_FromSize_t(v.index));
      PyList_SetItem(weights, index, PyFloat_FromDouble(v.weight));
      ++index;
    }
  return Py_BuildValue("NN", indexes, weights);
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
  ray_mean(double theta, double xi, size_t nx, size_t ny, double widthx,
      double widthy, double centerx, double centery, double xi_width,
      size_t xi_n)
  {
    const double pitchx = widthx / nx;
    const double pitchy = widthy / ny;
    const double weight_limit = 1e-12
        * std::sqrt(pitchx * pitchx + pitchy * pitchy);

    if (xi_width > 0 && xi_n > 1)
      {
        const double xi_pitch = xi_width / (xi_n - 1);
        const double xi_start = xi - .5 * xi_width;

        ray_type ray;
        for (size_t i = 0; i < xi_n; ++i)
          {
            ray_type ray_new = ray_func(theta, xi_start + i * xi_pitch, nx, ny,
                widthx, widthy, centerx, centery);

            for (auto& v : ray)
              {
                auto it = find_if(ray_new.begin(), ray_new.end(),
                    [v](const Vertex& v_new)
                      { return v.index==v_new.index;});
                if (it != ray_new.end())
                  {
                    v.weight += it->weight;
                    ray_new.erase(it);
                  }
              }
            ray.splice(ray.end(), ray_new);
          }
        const double norm = 1. / (xi_n);
        for (auto& v : ray)
          {
            v.weight *= norm;
          }
        ray.remove_if([weight_limit](const Vertex& v)
          { return v.weight < weight_limit;});
        ray.sort(cmp_less_index);
        return ray;
      }
    return ray_func(theta, xi, nx, ny, widthx, widthy, centerx, centery);
  }

template<ray_func_ptr ray_func>
  ray_type
  ray_diff(double theta, double xi, size_t n0, size_t n1, double width0,
      double width1, double center0, double center1, double xi_diff,
      double xi_mean, size_t xi_n)
  {
    auto ray_low = ray_mean<ray_func>(theta, xi - .5 * xi_diff, n0, n1, width0,
        width1, center0, center1, xi_mean, xi_n);
    auto ray_up = ray_mean<ray_func>(theta, xi + .5 * xi_diff, n0, n1, width0,
        width1, center0, center1, xi_mean, xi_n);
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
    const double norm = 1. / xi_diff;
    for (auto v : ray_up)
      {
        v.weight *= norm;
      }
    ray_up.remove_if([weight_limit](const Vertex& v)
      { return std::abs(v.weight)<weight_limit;});
    ray_up.sort(cmp_less_index);
    return ray_up;
  }

void
projector_mask_radial(projector_type& projector, size_t nx, size_t ny,
    double widthy, double widthx, double offsetx, double offsety, double radius)
{
  const auto rr = radius * radius;
  const double pitchx = widthx / (nx - 1);
  const double pitchy = widthx / (ny - 1);
  const double startx = -.5 * widthx + offsetx;
  const double starty = -.5 * widthy + offsety;
  auto out_of_radius = [rr, nx, pitchx, pitchy, startx, starty](const Vertex& v)
    {
      double x = startx + size_t(v.index%nx)*pitchx;
      double y = starty + size_t(v.index/nx)*pitchy;
      return ((x*x+y*y) > rr );
    };
  for (auto& ray : projector)
    {
      ray.remove_if(out_of_radius);
    }
}

template<ray_func_ptr ray_func, class Iterator>
  projector_type
  get_projector(Iterator thetas_begin, const Iterator thetas_end,
      Iterator xis_begin, const Iterator xis_end, size_t n, int nx, int ny,
      double widthx, double widthy, double centerx, double centery,
      double xi_diff, double xi_mean, size_t xi_n, double mask_radius)
  {
    projector_type projector;
    projector.reserve(n);
    const bool bdiff = (xi_diff != 0);
    for (auto thetas_it = thetas_begin; thetas_it != thetas_end; ++thetas_it)
      {
        for (auto xis_it = xis_begin; xis_it != xis_end; ++xis_it)
          {
            if (bdiff)
              {
                projector.push_back(
                    ray_diff<ray_func>(*thetas_it, *xis_it, nx, ny, widthx,
                        widthy, centerx, centery, xi_diff, xi_mean, xi_n));
              }
            else
              {
                projector.push_back(
                    ray_mean<ray_func>(*thetas_it, *xis_it, nx, ny, widthx,
                        widthy, centerx, centery, xi_mean, xi_n));
              }
          }
      }
    if (mask_radius != 0)
      {
        projector_mask_radial(projector, nx, ny, widthx, widthy, 0., 0.,
            mask_radius);
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
    const size_t size_i =
        (self->_shape_i->size() == 0) ?
            0 :
            std::accumulate(self->_shape_i->begin(), self->_shape_i->end(), 1,
                std::multiplies<size_t>());
    _fp_Projector* ret = reinterpret_cast<_fp_Projector*>(PyObject_CallObject(
        reinterpret_cast<PyObject*>(self->ob_type), nullptr));

    ret->_projector->resize(size_i);
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
        //PyErr_SetString(PyExc_ValueError, "");
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
    return reinterpret_cast<PyObject*>(arr_projection.new_ref());
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
  double xi_diff = 0.;
  double xi_mean = 0.;
  int xi_n = 0;

  static const char* kwlist[] =
    { "theta", "xi", "nx", "ny", "widthx", "widthy", "centerx", "centery",
        "xi_diff", "xi_mean", "xi_n", nullptr };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ddiidd|ddddi:siddon2d",
      const_cast<char**>(kwlist), &theta, &xi, &nx, &ny, &widthx, &widthy,
      &centerx, &centery, &xi_diff, &xi_mean, &xi_n))
    {
      return nullptr;
    }
  if (xi_diff != 0.) // xi_diff is set.
    {
      return ray_to_PyObject(
          ray_diff<siddon2d>(theta, xi, nx, ny, widthx, widthy, centerx,
              centery, xi_diff, xi_mean, xi_n));
    }
  return ray_to_PyObject(
      ray_mean<siddon2d>(theta, xi, nx, ny, widthx, widthy, centerx, centery,
          xi_mean, xi_n));
}

static PyObject*
_fp_projector_siddon2d(PyObject* self, PyObject* args, PyObject* kwargs)
{
  PyObject* thetas = nullptr;
  PyObject* xis = nullptr;
  int nx;
  int ny;
  double widthx;
  double widthy;
  double centerx = 0.;
  double centery = 0.;
  double xi_diff = 0.;
  double xi_mean = 0.;
  int xi_n = 0;
  double mask_radius = 0.;

  static const char* kwlist[] =
    { "thetas", "xis", "nx", "ny", "widthx", "widthy", "centerx", "centery",
        "xi_diff", "xi_mean", "xi_n", "mask_radius", nullptr };

  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
      "OOiidd|ddddid:projector_siddon2d", const_cast<char**>(kwlist), &thetas,
      &xis, &nx, &ny, &widthx, &widthy, &centerx, &centery, &xi_diff, &xi_mean,
      &xi_n, &mask_radius))
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

  *(ret->_projector) = get_projector<siddon2d>(arr_thetas.begin(),
      arr_thetas.end(), arr_xis.begin(), arr_xis.end(), n, nx, ny, widthx,
      widthy, centerx, centery, xi_diff, xi_mean, xi_n, mask_radius);

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
