ferret-opencv
=============

Ferret bindings for OpenCV

    (require '[ferret-opencv.core :as cv])

    (def cam (cv/video-capture 0))

    (let [f (cv/query-capture cam)]
      (cv/imwrite "image_latest.png" f))

Sample CMake,

    cmake_minimum_required(VERSION 3.0)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(OpenCV REQUIRED)

    add_executable(core core.cpp)

    target_link_libraries(core ${OpenCV_LIBS})
