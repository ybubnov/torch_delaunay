#include <torchgis/cartesian.h>

#include <pybind11/pybind11.h>


namespace torchgis {
namespace cartesian {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circumradius2d", &circumradius2d);

  m.def("circumcenter2d", &circumcenter2d);
};


} // cartesian
} // torchgis
