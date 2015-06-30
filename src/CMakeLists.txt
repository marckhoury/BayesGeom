cmake_minimum_required(VERSION 2.8)
project(predicates)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(boost-python.cmake)

add_library(c_predicates SHARED predicates.h predicates.c)

include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
boost_python_module(predicates pyPredicates.cpp)
target_link_libraries(predicates c_predicates)