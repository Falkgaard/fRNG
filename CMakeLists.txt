#pragma once
cmake_minimum_required ( VERSION 3.21 FATAL_ERROR ) # TODO: expand version range

project (
	fRNG
	VERSION     0.1
	DESCRIPTION "A FALK library module for pseudo-random number generation."
	LANGUAGES   CXX
)

configure_file ( # for exposing info to C++ (e.g. project version)
	"${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.hh.in"
	"${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.hh"
)

set          ( CMAKE_POSITION_INDEPENDENT_CODE     ON ) # -fPIC
set          ( CMAKE_INTERPROCEDURAL_OPTIMIZATION  ON )
cmake_policy ( SET                  CMP0069       NEW )
set          ( CMAKE_POLICY_DEFAULT_CMP0069       NEW )

add_subdirectory           ( include/${PROJECT_NAME} )
add_library                ( ${PROJECT_NAME} INTERFACE )
target_compile_features    ( ${PROJECT_NAME} INTERFACE cxx_std_23 )
target_include_directories ( ${PROJECT_NAME} INTERFACE include/ )

set ( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include ( cmake/CPM.cmake )
	 CPMAddPackage( "gh:falkgaard/fHash@0.1" )
target_link_libraries ( ${PROJECT_NAME} INTERFACE fHash )
