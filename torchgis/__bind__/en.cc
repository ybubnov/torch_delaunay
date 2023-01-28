#include <torchgis/torchgis.h>

#include <pybind11/pybind11.h>


namespace torchgis {
namespace en {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("circumradius2d", &circumradius2d);

  m.def("circumcenter2d", &circumcenter2d);

  m.def("shull2d", &shull2d);
};


} // en
} // torchgis
