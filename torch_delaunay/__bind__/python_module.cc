/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
