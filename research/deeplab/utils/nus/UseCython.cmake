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
