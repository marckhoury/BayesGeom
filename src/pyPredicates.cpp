#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include "predicates.h"

extern "C" {

using namespace std;
namespace py = boost::python;

REAL py_orient2d(py::object py_pa, py::object py_pb, py::object py_pc){
  REAL* pa = getPointer<REAL>(py_pa);
  REAL* pb = getPointer<REAL>(py_pb);
  REAL* pc = getPointer<REAL>(py_pc);
  REAL res = orient2d(pa, pb, pc);
  return res;
}

REAL py_orient3d(py::object py_pa, py::object py_pb, py::object py_pc, py::object py_pd){
  REAL* pa = getPointer<REAL>(py_pa);
  REAL* pb = getPointer<REAL>(py_pb);
  REAL* pc = getPointer<REAL>(py_pc);
  REAL* pd = getPointer<REAL>(py_pd);
  REAL res = orient3d(pa, pb, pc, pd);
  return res;
}

REAL py_incircle(py::object py_pa, py::object py_pb, py::object py_pc, py::object py_pd){
  REAL* pa = getPointer<REAL>(py_pa);
  REAL* pb = getPointer<REAL>(py_pb);
  REAL* pc = getPointer<REAL>(py_pc);
  REAL* pd = getPointer<REAL>(py_pd);
  REAL res = incircle(pa, pb, pc, pd);
  return res;
}

REAL py_insphere(py::object py_pa, py::object py_pb, py::object py_pc, py::object py_pd){
  REAL* pa = getPointer<REAL>(py_pa);
  REAL* pb = getPointer<REAL>(py_pb);
  REAL* pc = getPointer<REAL>(py_pc);
  REAL* pd = getPointer<REAL>(py_pd);
  REAL res = insphere(pa, pb, pc, pd);
  return res;
}


BOOST_PYTHON_MODULE(predicates) {
    np_mod = py::import("numpy");
    exactinit(); // run the initialization when we import?
    py::def("orient2d", &py_orient2d, (py::arg("py_pa"), py::arg("py_pb"), py::arg("py_pc")));
    py::def("orient3d", &py_orient3d, (py::arg("py_pa"), py::arg("py_pb"), py::arg("py_pc"), py::arg("py_pd")));
    py::def("incircle", &py_incircle, (py::arg("py_pa"), py::arg("py_pb"), py::arg("py_pc"), py::arg("py_pd")));
    py::def("insphere", &py_insphere, (py::arg("py_pa"), py::arg("py_pb"), py::arg("py_pc"), py::arg("py_pd")));
}

}
