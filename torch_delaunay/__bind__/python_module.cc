// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#include <torch_delaunay/torch_delaunay.h>

#include <pybind11/pybind11.h>


namespace torch_delaunay {


using tensor_ref = const torch::Tensor&;
using circum_decltype = torch::Tensor (*)(tensor_ref, tensor_ref, tensor_ref);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("shull2d", &shull2d);
    m.def("circumcenter2d", static_cast<circum_decltype>(&circumcenter2d));
    m.def("circumradius2d", static_cast<circum_decltype>(&circumradius2d));
};


} // namespace torch_delaunay
