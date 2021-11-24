cmake_minimum_required( VERSION 3.21 FATAL_ERROR )

project(
	fRNG
	VERSION     0.1.0
	DESCRIPTION "A FALK library module for pseudo-random number generation."
	LANGUAGES   CXX
)

set          ( CMAKE_POSITION_INDEPENDENT_CODE     ON ) # -fPIC
set          ( CMAKE_INTERPROCEDURAL_OPTIMIZATION  ON )
cmake_policy ( SET                  CMP0069       NEW )
set          ( CMAKE_POLICY_DEFAULT_CMP0069       NEW )

include                    ( GNUInstallDirs )
add_library                ( ${PROJECT_NAME} INTERFACE )
target_compile_features    ( ${PROJECT_NAME} INTERFACE cxx_std_23 )
target_include_directories ( ${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

configure_file ( # for exposing info to C++ (e.g. project version)
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/info.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.h
)

# OLD (TODO: remove?)
# set ( HEADERS
# 	core.h
# 	info.h
# )
# add_library                ( ${PROJECT_NAME} INTERFACE ) 
# target_sources             ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/${HEADERS} )
# target_include_directories ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )

#---------------------------------- Testing: ----------------------------------#

# TODO: doctest
# find_package( Catch2 REQUIRED )
# add_executable( fRNG_test test/test.cc )
# target_link_libraries( fRNG_test PRIVATE Catch2::Catch2 )
# target_include_directories( fRNG_test PRIVATE ${PROJECT_SOURCE_DIR}/include )
# include( CTest )
# include( Catch )
# catch_discover_tests( fRNG_test )
# enable_testing()

#--------------------------- External dependencies: ----------------------------#

set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
include( cmake/CPM.cmake )
	CPMAddPackage( "gh:falkgaard/fHash@0.1" )
target_link_libraries( fRNG INTERFACE fHash::fHash ) # <- add library dependency targets here

#--------------------------------- Install: ------------------------------------#

install(
	TARGETS ${PROJECT_NAME}
	EXPORT ${PROJECT_NAME}_targets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include( CMakePackageConfigHelpers )

write_basic_package_version_file(
	"${PROJECT_NAME}_config_version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

if ( NOT INCLUDE_INSTALL_DIR )
	set ( INCLUDE_INSTALL_DIR include/${PROJECT_NAME} )
endif()

configure_package_config_file(
	"${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake.in"
	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.cmake"
	INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
	PATH_VARS INCLUDE_INSTALL_DIR
)

install( EXPORT ${PROJECT_NAME}_targets
	FILE         ${PROJECT_NAME}_targets.cmake
	NAMESPACE    ${PROJECT_NAME}::
	DESTINATION  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
	FILES       "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.cmake"
	            "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config_version.cmake"
	DESTINATION  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(
	DIRECTORY    ${PROJECT_SOURCE_DIR}/include/
	DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# OLD (TODO: remove?)
# target_compile_features ( ${PROJECT_NAME} PUBLIC cxx_std_23                                        ) # Set C++ standard to C++2b
# target_compile_options  ( ${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->" ) # To help ensure cross-platform compatibility
# set_target_properties   ( ${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF                            ) # Ditto
