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

set ( HEADERS
	core.h
	info.h
)
# OLD # include                    ( GNUInstallDirs )
add_library                ( ${PROJECT_NAME} INTERFACE )
target_compile_features    ( ${PROJECT_NAME} INTERFACE cxx_std_23 )
target_sources             ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/${HEADERS} )
target_include_directories ( ${PROJECT_NAME} include/ )
# OLD # target_include_directories ( ${PROJECT_NAME} INTERFACE
# OLD #     $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
# OLD #     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# OLD # )
# OLD # # OLD: target_include_directories ( ${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include )

configure_file ( # for exposing info to C++ (e.g. project version)
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/info.h.in
	${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/info.h
)

# OLD # #---------------------------------- Testing: ----------------------------------#
# OLD # 
# OLD # # TODO: switch to doctest instead
# OLD # # find_package( Catch2 REQUIRED )
# OLD # # add_executable( fRNG_test test/test.cc )
# OLD # # target_link_libraries( fRNG_test PRIVATE Catch2::Catch2 )
# OLD # # target_include_directories( fRNG_test PRIVATE ${PROJECT_SOURCE_DIR}/include )
# OLD # # include( CTest )
# OLD # # include( Catch )
# OLD # # catch_discover_tests( fRNG_test )
# OLD # # enable_testing()
# OLD # 
# OLD # #--------------------------- External dependencies: ----------------------------#
# OLD # 
# OLD # set( CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )
# OLD # include( cmake/CPM.cmake )
# OLD # 	CPMAddPackage( "gh:falkgaard/fHash@0.1" )
# OLD # target_link_libraries( fRNG INTERFACE fHash::fHash ) # <- add library dependency targets here
# OLD # 
# OLD # #--------------------------------- Install: ------------------------------------#
# OLD # 
# OLD # install(
# OLD # 	TARGETS ${PROJECT_NAME}
# OLD # 	EXPORT ${PROJECT_NAME}_targets
# OLD # 	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# OLD # 	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
# OLD # 	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# OLD # )
# OLD # 
# OLD # include( CMakePackageConfigHelpers )
# OLD # 
# OLD # write_basic_package_version_file(
# OLD # 	"${PROJECT_NAME}_config_version.cmake"
# OLD # 	VERSION ${PROJECT_VERSION}
# OLD # 	COMPATIBILITY SameMajorVersion
# OLD # )
# OLD # 
# OLD # if ( NOT INCLUDE_INSTALL_DIR )
# OLD # 	set ( INCLUDE_INSTALL_DIR include/${PROJECT_NAME} )
# OLD # endif()
# OLD # 
# OLD # configure_package_config_file(
# OLD # 	"${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}_config.cmake.in"
# OLD # 	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.cmake"
# OLD # 	INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
# OLD # 	PATH_VARS INCLUDE_INSTALL_DIR
# OLD # )
# OLD # 
# OLD # install( EXPORT ${PROJECT_NAME}_targets
# OLD # 	FILE         ${PROJECT_NAME}_targets.cmake
# OLD # 	NAMESPACE    ${PROJECT_NAME}::
# OLD # 	DESTINATION  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
# OLD # )
# OLD # 
# OLD # install(
# OLD # 	FILES       "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config.cmake"
# OLD # 	            "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_config_version.cmake"
# OLD # 	DESTINATION  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
# OLD # )
# OLD # 
# OLD # install(
# OLD # 	DIRECTORY    ${PROJECT_SOURCE_DIR}/include/
# OLD # 	DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
# OLD # )
# OLD # 
# OLD # # OLD (TODO: remove?)
# OLD # # target_compile_options  ( ${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->" ) # To help ensure cross-platform compatibility
# OLD # # set_target_properties   ( ${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF                            ) # Ditto
