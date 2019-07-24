#include "closest_point.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

template<int Dim>
struct D {
  template<class T, class R>
  static void diff(const T* x, const R* y, T* z, double w) {
    static_assert(Dim>0, "Dim>0!");
    z[0] = w * (x[0] - y[0]);
    D<Dim - 1>::diff(x + 1, y + 1, z + 1, w);
  }
};


template<>
struct D<0> {
  template<class T, class R>
  static void diff(const T* x, const R* y, T* z, double w) { }
};


template<int Dim>
struct UnaryDiff {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    D<Dim>::diff(x, ptr, residual, weight);
    return true;
  }

  const double* ptr;
  double weight;
};


template<int Dim>
struct PairwiseDiff {
  template <typename T>
  bool operator()(const T* const x, const T* const y, T* residual) const {
    D<Dim>::diff(x, y, residual, weight);
    return true;
  }
  double weight;
};


void closest_point(const double* target, int h, int w, double weight, double* out) {
  double w_u = sqrt(weight);
  double w_p = sqrt(1 - w_u);
  Problem problem;

  // energy terms
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j) {
      // distance to boundary
      CostFunction* cost_function =
          new AutoDiffCostFunction<UnaryDiff<2>, 2, 2>(
            new UnaryDiff<2> {target + (i * w + j) * 2, w_u});
      problem.AddResidualBlock(cost_function, NULL, out + (i * w + j) * 2);

      // distance to neighbours
      if (i + 1 < h) {
        cost_function =
            new AutoDiffCostFunction<PairwiseDiff<2>, 2, 2, 2>(
              new PairwiseDiff<2>{w_p});
        problem.AddResidualBlock(cost_function, NULL,
          out + (i * w + j) * 2, out + ((i + 1) * w + j) * 2);
      }
      if (j + 1 < w) {
        cost_function =
            new AutoDiffCostFunction<PairwiseDiff<2>, 2, 2, 2>(
              new PairwiseDiff<2>{w_p});
        problem.AddResidualBlock(cost_function, NULL,
          out + (i * w + j) * 2, out + (i * w + j + 1) * 2);
      }
    }

  // boundary conditions
  if (w > 2) {
    ceres::SubsetParameterization *fix0 =
      new ceres::SubsetParameterization(2, {0});
    for (int i : {0, h - 1})
      for (int j = 1; j + 1 < w; ++j) {
        out[(i * w + j) * 2] = i / (h - 1);
        problem.SetParameterization(out + (i * w + j) * 2, fix0);
      }
  }
  if (h > 2) {
    ceres::SubsetParameterization *fix1 =
      new ceres::SubsetParameterization(2, {1});
    for (int i = 1; i + 1 < h; ++i)
      for (int j : {0, w - 1}) {
        out[(i * w + j) * 2 + 1] = j / (w - 1);
        problem.SetParameterization(out + (i * w + j) * 2, fix1);
      }
    }

  for (int i : {0, h - 1})
    for (int j : {0, w - 1}) {
      out[(i * w + j) * 2] = i / (h - 1);
      out[(i * w + j) * 2 + 1] = j / (w - 1);
      problem.SetParameterBlockConstant(out + (i * w + j) * 2);
    }

  Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
}
