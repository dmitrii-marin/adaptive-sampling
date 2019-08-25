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

# This allows to link Cython files
# Examples:
# 1) to compile assembly.pyx to assembly.so:
#   CYTHON_ADD_MODULE(assembly)
# 2) to compile assembly.pyx and something.cpp to assembly.so:
#   CYTHON_ADD_MODULE(assembly something.cpp)

find_program(CYTHON NAMES cython cython.py)

if(NOT CYTHON_INCLUDE_DIRECTORIES)
    set(CYTHON_INCLUDE_DIRECTORIES .)
endif(NOT CYTHON_INCLUDE_DIRECTORIES)

macro(CYTHON_ADD_MODULE name)

    add_custom_command(
        OUTPUT ${name}.cpp
        COMMAND ${CYTHON}
        ARGS -3 -I ${CYTHON_INCLUDE_DIRECTORIES} -o ${name}.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${name}.pyx
        DEPENDS ${name}.pyx
        COMMENT "Cython source")

    add_library(${name} MODULE ${name}.cpp ${ARGN})

    set_target_properties(${name} PROPERTIES PREFIX "")
	if (CMAKE_HOST_WIN32)
		set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
	endif(CMAKE_HOST_WIN32)

endmacro(CYTHON_ADD_MODULE)

macro(CYTHON_COPY name)

	get_target_property(FILEPATH ${name} LOCATION)

	add_custom_command(TARGET ${name} POST_BUILD
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy ${FILEPATH} ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Copy python module to source dir"
	)

endmacro(CYTHON_COPY)
