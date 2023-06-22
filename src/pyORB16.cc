#include "ORB16.h"
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int toOpenCVType(const py::dtype &dtype) {
  if (dtype.is(py::dtype::of<uint8_t>()))
    return CV_8UC1;
  else if (dtype.is(py::dtype::of<uint16_t>()))
    return CV_16UC1;
  else if (dtype.is(py::dtype::of<float>()))
    return CV_32FC1;
  else if (dtype.is(py::dtype::of<double>()))
    return CV_64FC1;
  else
    throw std::runtime_error("Unsupported type passed to toMat");
}

template <typename T> cv::Mat toMat(const py::array_t<T> &input) {
  if (input.ndim() != 2)
    throw std::runtime_error("Number of dimensions must be two");
  int type = toOpenCVType(input.dtype());
  auto buf = input.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], type, (void *)buf.ptr);
  return mat;
}

template <typename T> py::array_t<T> toArray(const cv::Mat &input) {
  // input cv::Mat -> py::buffer
  py::buffer_info buffer_info_descriptors =
      py::buffer_info(input.data, sizeof(T), py::format_descriptor<T>::format(), 2,
                      {input.rows, input.cols}, {input.step[0], input.step[1]});
  // py::buffer -> py::array
  return py::array_t<T>(buffer_info_descriptors);
}

template <typename T> void definePointClass(py::module &m, const std::string &name) {
  using class_ = cv::Point_<T>;
  py::class_<class_>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<T, T>())
      .def_readwrite("x", &class_::x)
      .def_readwrite("y", &class_::y)
      .def("__repr__",
           [](const class_ &p) {
             return "(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
           })
      // make it subscriptable
      .def("__getitem__",
           [](const class_ &p, int i) {
             if (i == 0)
               return p.x;
             else if (i == 1)
               return p.y;
             else
               throw py::index_error();
           })
      .def("__setitem__", [](class_ &p, int i, T v) {
        if (i == 0)
          p.x = v;
        else if (i == 1)
          p.y = v;
        else
          throw py::index_error();
      });
}

cv::ORB16 ORB16_create(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8,
                       int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2,
                       cv::ORB16::ScoreType scoreType = cv::ORB16::ScoreType::HARRIS_SCORE,
                       int patchSize = 31, int fastThreshold = 20) {
  return cv::ORB16(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType,
                   patchSize, fastThreshold);
}

PYBIND11_MODULE(pyORB16, m) {
  m.doc() = "ORB16 class implementation"; // Optional module docstring

  py::class_<cv::KeyPoint>(m, "KeyPoint", py::dynamic_attr())
      .def(py::init<>())
      .def(py::init<float, float, float, float, int, int, int>())
      .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id);

  definePointClass<int>(m, "Point2i");
  definePointClass<float>(m, "Point2f");

  py::enum_<cv::ORB16::ScoreType>(m, "ScoreType")
      .value("HARRIS_SCORE", cv::ORB16::ScoreType::HARRIS_SCORE)
      .value("FAST_SCORE", cv::ORB16::ScoreType::FAST_SCORE);

  py::class_<cv::ORB16>(m, "ORB16")
      .def(py::init<int, float, int, int, int, int, cv::ORB16::ScoreType, int, int>(),
           py::arg("nfeatures") = 500, py::arg("scaleFactor") = 1.2f, py::arg("nlevels") = 8,
           py::arg("edgeThreshold") = 31, py::arg("firstLevel") = 0, py::arg("WTA_K") = 2,
           py::arg("scoreType") = cv::ORB16::ScoreType::HARRIS_SCORE, py::arg("patchSize") = 31,
           py::arg("fastThreshold") = 20)
      .def("setMaxFeatures", &cv::ORB16::setMaxFeatures)
      .def("getMaxFeatures", &cv::ORB16::getMaxFeatures)
      .def("setScaleFactor", &cv::ORB16::setScaleFactor)
      .def("getScaleFactor", &cv::ORB16::getScaleFactor)
      .def("setNLevels", &cv::ORB16::setNLevels)
      .def("getNLevels", &cv::ORB16::getNLevels)
      .def("setEdgeThreshold", &cv::ORB16::setEdgeThreshold)
      .def("getEdgeThreshold", &cv::ORB16::getEdgeThreshold)
      .def("setFirstLevel", &cv::ORB16::setFirstLevel)
      .def("getFirstLevel", &cv::ORB16::getFirstLevel)
      .def("setWTA_K", &cv::ORB16::setWTA_K)
      .def("getWTA_K", &cv::ORB16::getWTA_K)
      .def("setScoreType", &cv::ORB16::setScoreType)
      .def("getScoreType", &cv::ORB16::getScoreType)
      .def("setPatchSize", &cv::ORB16::setPatchSize)
      .def("getPatchSize", &cv::ORB16::getPatchSize)
      .def("setFastThreshold", &cv::ORB16::setFastThreshold)
      .def("getFastThreshold", &cv::ORB16::getFastThreshold)
      .def("getDefaultName", &cv::ORB16::getDefaultName)
      .def("descriptorSize", &cv::ORB16::descriptorSize)
      .def("descriptorType", &cv::ORB16::descriptorType)
      .def("defaultNorm", &cv::ORB16::defaultNorm)
      .def(
          "detect",
          [](cv::ORB16 &self, py::array_t<uint16_t> py_array, py::object py_mask = py::none()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat image = toMat<uint16_t>(py_array);
            cv::Mat mask = py::isinstance<py::none>(py_mask) ?
                            cv::Mat() :
                            toMat<uint8_t>(py_mask.cast<py::array_t<uint8_t>>());
            self.detect(image, keypoints, mask);
            return keypoints;
          },
          py::arg("image"), py::arg("mask"))
      .def(
          "compute",
          [](cv::ORB16 &self, py::array_t<uint16_t> py_array, std::vector<cv::KeyPoint> keypoints) {
            cv::Mat descriptors;
            cv::Mat image = toMat<uint16_t>(py_array);
            self.compute(image, keypoints, descriptors);
            py::array_t<uint8_t> py_descriptors = toArray<uint8_t>(descriptors);
            return std::make_tuple(keypoints, py_descriptors);
          },
          py::arg("image"), py::arg("keypoints"))
      .def(
          "detectAndCompute",
          [](cv::ORB16 &self, py::array_t<uint16_t> py_array, py::object py_mask = py::none()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            cv::Mat image = toMat<uint16_t>(py_array);
            cv::Mat mask = py::isinstance<py::none>(py_mask) ?
                            cv::Mat() :
                            toMat<uint8_t>(py_mask.cast<py::array_t<uint8_t>>());
            self.detectAndCompute(image, mask, keypoints, descriptors);
            py::array_t<uint8_t> py_descriptors = toArray<uint8_t>(descriptors);
            return std::make_tuple(keypoints, py_descriptors);
          },
          py::arg("image"), py::arg("mask"));

  m.def("ORB16_create", &ORB16_create, py::arg("nfeatures") = 500, py::arg("scaleFactor") = 1.2f,
        py::arg("nlevels") = 8, py::arg("edgeThreshold") = 31, py::arg("firstLevel") = 0,
        py::arg("WTA_K") = 2, py::arg("scoreType") = cv::ORB16::ScoreType::HARRIS_SCORE,
        py::arg("patchSize") = 31, py::arg("fastThreshold") = 20);
}
