cmake_minimum_required(VERSION 2.18...3.16)
project(ORB16)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Release)
    message("Release Mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Mode")
    add_definitions(-DDEBUG -DDEBUGMSF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
endif ()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE
src/ORB16.cc
src/fast16.cc
src/canny16.cc
    )

add_library(${PROJECT_NAME} SHARED ${SOURCE})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

option(BUILD_TEST "Build test" OFF)
option(BUILD_PYTHON "Build pybind" OFF)

IF(BUILD_TEST)
    add_executable(test_ORB16 src/test_ORB16.cc)
    target_link_libraries(test_ORB16 ${PROJECT_NAME})

    add_executable(test_fast16 src/test_fast16.cc)
    target_link_libraries(test_fast16 ${PROJECT_NAME})

    add_executable(test_detection src/test_detection.cc)
    target_link_libraries(test_detection ${PROJECT_NAME})

    add_executable(test_canny16 src/test_canny16.cc)
    target_link_libraries(test_canny16 ${PROJECT_NAME})
ENDIF(BUILD_TEST)

IF(BUILD_PYTHON)
    add_subdirectory(python)
ENDIF(BUILD_PYTHON)
