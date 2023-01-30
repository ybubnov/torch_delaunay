#include <torchgis/torchgis.h>

#include <pybind11/pybind11.h>


namespace torchgis {
namespace en {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triangle_circumradius2d", &triangle_circumradius2d);

  m.def("triangle_circumcenter2d", &triangle_circumcenter2d);

  m.def("dist", &dist);

  m.def("shull2d", &shull2d);
};


} // en
} // torchgis
