#pragma once

#include <torch/torch.h>


namespace torchgis {
namespace en {

/*
template <typename scalar_t, typename index_t, typename combine_t>
struct LawsonFlipData {
  C10_HOST_DEVICE LawsonFlipData() {}
};


template <typename scalar_t, typename acc_scalar_t, typename index_t, typename combine_t, typename res_t>
struct LawsonFlipData {
 public:

  using acc_t = LawsonFlipData<scalar_t, index_t, combine_t>;

  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, index_t idx) const {
    return LawsonFlipData();
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return LawsonFlipData();
  }
};
*/

} // namespace en
} // namespace torchgis
