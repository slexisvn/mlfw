cmake_minimum_required(VERSION 3.20.0)
project(tera-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(EXTERNAL_PROJECT_BUILD FALSE)
  find_package(MLIR REQUIRED CONFIG)

  # Define the default argument to use by `lit` when testing.
  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  set(EXTERNAL_PROJECT_BUILD TRUE)
  # Build via external projects mechanism
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/tools/Tera/lib)

  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_MAIN_SRC_DIR}/cmake/modules")
endif()

set(TERA_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TERA_BINARY_DIR ${PROJECT_BINARY_DIR})
include(${TERA_SOURCE_DIR}/cmake/AddTera.cmake)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TERA_SOURCE_DIR}/include)
include_directories(${TERA_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
if(MLIR_INCLUDE_TESTS)
  add_subdirectory(test)
endif()
add_subdirectory(tera-gradcheck)
add_subdirectory(tera-opt)
add_subdirectory(tera-runner)

include(GNUInstallDirs)

install(DIRECTORY ${TERA_SOURCE_DIR}/include/Tera
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT tera-headers
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.td"
        )

install(DIRECTORY ${TERA_BINARY_DIR}/include/Tera
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT tera-headers
        FILES_MATCHING
        PATTERN "*.inc"
        PATTERN "CMakeFiles" EXCLUDE
        )

include(CMakePackageConfigHelpers)
configure_package_config_file(
        ${TERA_SOURCE_DIR}/cmake/TeraConfig.cmake.in
        ${TERA_BINARY_DIR}/cmake/TeraConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tera)

install(EXPORT MLIRTargets
        FILE TeraTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tera
        COMPONENT tera-cmake-exports)

install(FILES ${TERA_BINARY_DIR}/cmake/TeraConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tera
        COMPONENT tera-cmake-exports)
