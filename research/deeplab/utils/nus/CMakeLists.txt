# Copyright 2019 Dmitrii Marin (https://github.com/dmitrii-marin) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
