add_subdirectory(pybind11)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${OpenCV_INCLUDE_DIRS}
)

pybind11_add_module(pyORB16 pyORB16.cc)
target_link_libraries(pyORB16 PRIVATE ORB16)
