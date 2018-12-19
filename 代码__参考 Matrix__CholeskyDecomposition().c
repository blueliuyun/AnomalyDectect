/**
 * 参考地址
 * https://github.com/cognitoware/robotics/blob/4d99769c6845008559e19dbcb718a8b53f233f9c/cognitoware/math/data/Matrix.cc
 */ 
Matrix Matrix::CholeskyDecomposition() const {
  // returns an upper diagonal matrix
  if (rows_ != cols_) {
    throw std::runtime_error("Matrix must be square.");
  }
  if (!IsSymmetric()) {
    throw std::runtime_error("Matrix must be symmetric.");
  }
  std::size_t n = rows();
  std::vector<double> a(m_);
  for (std::size_t i = 0; i < n; i++) {
    std::size_t ii = GetIndex(i, i);
    for (std::size_t k = 0; k < i; k++) {
      std::size_t ki = GetIndex(k, i);
      a[ii] = a[ii] - a[ki] * a[ki];
    }
    if (a[ii] < 0) {
      throw std::runtime_error("Matrix is not positive definite.");
    }
    a[ii] = sqrt(a[ii]);
    for (std::size_t j = i + 1; j < n; j++) {
      std::size_t ij = GetIndex(i, j);
      for (std::size_t k = 0; k < i; k++) {
        std::size_t ki = GetIndex(k, i);
        std::size_t kj = GetIndex(k, j);
        a[ij] = a[ij] - a[ki] * a[kj];
      }
      if (a[ij] != 0) a[ij] = a[ij] / a[ii];
    }
  }
  // Clear out the lower matrix
  for (std::size_t i = 1; i < n; i++) {
    for (std::size_t j = 0; j < i; j++) {
      std::size_t ij = GetIndex(i, j);
      a[ij] = 0;
    }
  }
  return Matrix(n, n, a);
}
