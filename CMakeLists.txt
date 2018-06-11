cmake_minimum_required(VERSION 2.8.12)

project(parallel_project)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

FIND_PACKAGE( PackageHandleStandardArgs )

SET (OPENCL_VERSION_STRING "0.1.0")
SET (OPENCL_VERSION_MAJOR 0)
SET (OPENCL_VERSION_MINOR 1)
SET (OPENCL_VERSION_PATCH 0)

FIND_PACKAGE_HANDLE_STANDARD_ARGS( OpenCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS )

IF( _OPENCL_CPP_INCLUDE_DIRS )
        SET( OPENCL_HAS_CPP_BINDINGS TRUE )
        LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
        # This is often the same, so clean up
        LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
ENDIF( _OPENCL_CPP_INCLUDE_DIRS )

MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIRS
)

find_package( OpenCV REQUIRED )

find_package(OpenCL REQUIRED)

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 

LINK_DIRECTORIES(${OpenCL_LIBRARY})

add_executable(${PROJECT_NAME} main.cpp utils.cpp interpolation_cpu.cpp interpolation_gpu.cpp ocl_macros.h)

target_link_libraries(${PROJECT_NAME}  ${OPENCL_LIBRARIES} ${OpenCV_LIBS} )

configure_file(image.dat ${CMAKE_CURRENT_BINARY_DIR}/image.dat COPYONLY)

file(GLOB SRC_LIST kernel/*.cl)
foreach(cl_file ${SRC_LIST})
        STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${cl_file})
        configure_file(${cl_file} ${CMAKE_CURRENT_BINARY_DIR}/kernel/${FILE_NAME}.cl COPYONLY)
endforeach()
