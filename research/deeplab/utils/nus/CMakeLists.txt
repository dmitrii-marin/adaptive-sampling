cmake_minimum_required(VERSION 3.0)

set(CMAKE_INCLUDE_CURRENT_DIR on)
set(CMAKE_CXX_STANDARD 11)

include(UseCython.cmake)
find_package( Ceres REQUIRED )

set( PYTHON_INCLUDE_DIR ~/anaconda3/include/python3.6m)
find_package( PythonLibs 3 REQUIRED )

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})

CYTHON_ADD_MODULE(closest_point_cython  closest_point.cpp )
target_link_libraries(closest_point_cython ceres)
