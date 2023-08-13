add_subdirectory(pybind11)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${OpenCV_INCLUDE_DIRS}
)

pybind11_add_module(cv16 cv16.cc)
target_link_libraries(cv16 PRIVATE ORB16)
