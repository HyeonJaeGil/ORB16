#include "ORB16.h"
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void defineKeyPointClass(py::module &m) {
  py::class_<cv::KeyPoint>(m, "KeyPoint", py::dynamic_attr())
      .def(py::init<>())
      .def(py::init<float, float, float, float, int, int, int>())
      .def_readwrite("pt", &cv::KeyPoint::pt)
      .def_readwrite("size", &cv::KeyPoint::size)
      .def_readwrite("angle", &cv::KeyPoint::angle)
      .def_readwrite("response", &cv::KeyPoint::response)
      .def_readwrite("octave", &cv::KeyPoint::octave)
      .def_readwrite("class_id", &cv::KeyPoint::class_id);
}

template <typename T>
void definePointClass(py::module &m, const std::string &name) {
  using class_ = cv::Point_<T>;
  py::class_<class_>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<T, T>())
      .def_readwrite("x", &class_::x)
      .def_readwrite("y", &class_::y)
      .def("__repr__", [](const class_ &p) {
        return "(" + std::to_string(p.x) + ", " + std::to_string(p.y) + ")";
      });
}

cv::ORB16 ORB16_create(
    int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8,
    int edgeThreshold = 31, int firstLevel = 0, int WTA_K = 2,
    cv::ORB16::ScoreType scoreType = cv::ORB16::ScoreType::HARRIS_SCORE,
    int patchSize = 31, int fastThreshold = 20) {
  return cv::ORB16(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                   WTA_K, scoreType, patchSize, fastThreshold);
}

PYBIND11_MODULE(pyORB16, m) {
  m.doc() = "ORB16 class implementation"; // Optional module docstring

  defineKeyPointClass(m);
  //   py::class_<cv::KeyPoint>(m, "KeyPoint");
  definePointClass<int>(m, "Point2i");
  definePointClass<float>(m, "Point2f");

  py::enum_<cv::ORB16::ScoreType>(m, "ScoreType")
      .value("HARRIS_SCORE", cv::ORB16::ScoreType::HARRIS_SCORE)
      .value("FAST_SCORE", cv::ORB16::ScoreType::FAST_SCORE);

  py::class_<cv::ORB16>(m, "ORB16")
      .def(py::init<int, float, int, int, int, int, cv::ORB16::ScoreType, int,
                    int>(),
           py::arg("nfeatures") = 500, py::arg("scaleFactor") = 1.2f,
           py::arg("nlevels") = 8, py::arg("edgeThreshold") = 31,
           py::arg("firstLevel") = 0, py::arg("WTA_K") = 2,
           py::arg("scoreType") = cv::ORB16::ScoreType::HARRIS_SCORE,
           py::arg("patchSize") = 31, py::arg("fastThreshold") = 20)
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
          [](cv::ORB16 &self, py::array_t<uint16_t> py_array,
             py::object py_mask = py::none()) {
            std::vector<cv::KeyPoint> keypoints;
            auto buffer_info = py_array.request();
            cv::Mat image(buffer_info.shape[0], buffer_info.shape[1], CV_16UC1,
                          buffer_info.ptr);
            if (py::isinstance<py::none>(py_mask)) {
              self.detect(image, keypoints, cv::noArray());
            } else {
              auto mask_buffer = py_mask.cast<py::array_t<uint8_t>>().request();
              cv::Mat mask(mask_buffer.shape[0], mask_buffer.shape[1], CV_8UC1,
                           mask_buffer.ptr);
              self.detect(image, keypoints, mask);
            }
            // py::list py_keypoints;
            // for (auto &keypoint : keypoints) {
            //   py_keypoints.append(py::cast(keypoint));
            // }
            return keypoints;
          },
          py::arg("image"), py::arg("mask"))
      .def(
          "detectAndCompute",
          [](cv::ORB16 &self, py::array_t<uint16_t> py_array,
             py::object py_mask = py::none()) {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;

            auto buffer_info = py_array.request();
            cv::Mat image(buffer_info.shape[0], buffer_info.shape[1], CV_16UC1,
                          buffer_info.ptr);

            if (py::isinstance<py::none>(py_mask)) {
              self.detectAndCompute(image, cv::noArray(), keypoints,
                                    descriptors, false);

            } else {
              auto mask_buffer = py_mask.cast<py::array_t<uint8_t>>().request();
              cv::Mat mask(mask_buffer.shape[0], mask_buffer.shape[1], CV_8UC1,
                           mask_buffer.ptr);
              self.detectAndCompute(image, mask, keypoints, descriptors, false);
            }

            // py::list py_keypoints;
            // for (auto &keypoint : keypoints) {
            //   py_keypoints.append(py::cast(keypoint));
            // }

            // convert descriptors to py::buffer
            py::buffer_info buffer_info_descriptors =
                py::buffer_info(descriptors.data, sizeof(uint8_t),
                                py::format_descriptor<uint8_t>::format(), 2,
                                {descriptors.rows, descriptors.cols},
                                {descriptors.step[0], descriptors.step[1]});
            py::array_t<uint8_t> py_descriptors(buffer_info_descriptors);

            // return keypoints and descriptors
            return std::make_tuple(keypoints, py_descriptors);
          },
          py::arg("image"), py::arg("mask"));

  m.def("ORB16_create", &ORB16_create, py::arg("nfeatures") = 500,
        py::arg("scaleFactor") = 1.2f, py::arg("nlevels") = 8,
        py::arg("edgeThreshold") = 31, py::arg("firstLevel") = 0,
        py::arg("WTA_K") = 2,
        py::arg("scoreType") = cv::ORB16::ScoreType::HARRIS_SCORE,
        py::arg("patchSize") = 31, py::arg("fastThreshold") = 20);
}
