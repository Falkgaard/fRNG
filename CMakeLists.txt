cmake_minimum_required( VERSION 3.21 FATAL_ERROR )

project(
	fRNG
	VERSION     1.0
	DESCRIPTION "A pseudo-random number generation library."
	LANGUAGES   CXX
)

set          ( CMAKE_POSITION_INDEPENDENT_CODE     ON ) # -fPIC
set          ( CMAKE_INTERPROCEDURAL_OPTIMIZATION  ON )
cmake_policy ( SET                  CMP0069       NEW )
set          ( CMAKE_POLICY_DEFAULT_CMP0069       NEW )

set ( HEADERS
	core.hh
	info.hh
	# ^--- add sources here
)

add_library                ( ${PROJECT_NAME} INTERFACE ) 
target_sources             ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/${HEADERS} )
target_include_directories ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )

configure_file ( # for exporting info to C++ (e.g. project version)
	${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.hh.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.hh
)

# target_compile_features ( ${PROJECT_NAME} PUBLIC cxx_std_23                                        ) # Set C++ standard to C++2b
# target_compile_options  ( ${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->" ) # To help ensure cross-platform compatibility
# set_target_properties   ( ${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF                            ) # Ditto

#--------------------------- External dependencies: ----------------------------#
include ( cmake/CPM.cmake )
CPMAddPackage( "gh:falkgaard/fHash"  ) # TODO tag
target_link_libraries ( fRNG PRIVATE fHash::fHash ) # <- add library dependency targets here
