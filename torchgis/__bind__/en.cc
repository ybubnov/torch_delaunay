#include <torchgis/torchgis.h>

#include <pybind11/pybind11.h>


namespace torchgis {
namespace en {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("orient2d", &orient2d);
  m.def("incircle2d", &incircle2d);

  m.def("circumradius2d", &circumradius2d);

  m.def("circumcenter2d", &circumcenter2d);

  m.def("dist", &dist);

  m.def("lawson_flip", &lawson_flip);
  m.def("shull2d", &shull2d);
};


} // en
} // torchgis
