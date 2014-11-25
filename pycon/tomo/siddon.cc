// This file is part of pycon: A framework for image reconstruction in X-ray
// Talbot-Lau interferometry.
//
// Copyright (C)
// 2012-2014 André Ritter (andre.ritter@fau.de)
#include "fp.h"
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

ray_type
siddon_vertices(double lstart, double lpitch, size_t size)
{
  ray_type ray;
  double length = lstart;
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

ray_type
siddon2d(double theta, double xi, size_t n0, size_t n1,
    double width0, double width1, double center0,
    double center1)
{
  ray_type ray;

  const double s = std::sin(theta);
  const double c = std::cos(theta);
  const double cinv = 1. / c;
  const double sinv = 1. / s;
  const double pitch0 = width0 / n0;
  const double pitch1 = width1 / n1;
  const double start0 = -.5 * width0 + center0;
  const double start1 = -.5 * width1 + center1;

  if (!ray_within_bounds(c, s, cinv, sinv, xi, start0, start0 + width0, start1,
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
      const double lpitch0 = lambda_by_x(0., c, sinv, pitch0);
      const double lpitch1 = lambda_by_y(0., s, cinv, pitch1);

      const double lstart0 = lambda_by_x(xi, c, sinv, start0);
      const double lstart1 = lambda_by_y(xi, s, cinv, start1);

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

      double weight_limit = 1e-12
          * std::sqrt(pitch0 * pitch0 + pitch1 * pitch1);
      while (it0 != it0_end && it1 != it1_end)
        {
          double weight = 0.;
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
