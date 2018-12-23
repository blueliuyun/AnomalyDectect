Matrix Matrix::Inverse() const {
  if (rows_ != cols_) {
    throw std::runtime_error("Matrix must be square.");
  }
  // set up mean as identity
  Matrix result(rows_, cols_);
  for (std::size_t i = 0; i < cols_; i++) {
    result.m_[result.GetIndex(i, i)] = 1.0;
  }

  Matrix working(rows_, cols_, m_);  // creates a copy of m_
  // Eliminate L
  for (std::size_t col = 0; col < cols_; col++) {
    double diag = working.at(col, col);
    for (std::size_t row = col + 1; row < rows_; row++) {
      double target = working.at(row, col);
      double a = -target / diag;
      working.CombineRow(col, row, a);
      result.CombineRow(col, row, a);
    }
    working.ScaleRow(col, 1.0 / diag);
    result.ScaleRow(col, 1.0 / diag);
  }
  // Eliminate U
  for (std::size_t col = cols_ - 1; col >= 1; col--) {
    double diag = working.at(col, col);  // 1.0
    for (std::size_t row_plus_one = col; row_plus_one > 0; row_plus_one--) {
      std::size_t row = row_plus_one - 1;
      double target = working.at(row, col);
      double a = -target / diag;
      working.CombineRow(col, row, a);
      result.CombineRow(col, row, a);
    }
  }
  return result;
}