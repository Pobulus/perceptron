cmake_minimum_required(VERSION 3.14)

project(perceptron_impl VERSION 0.0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(thirdparty/rapidyaml)

add_executable(perceptron main.cpp)
target_include_directories(perceptron PRIVATE thirdparty/rapidyaml/src)
target_link_libraries(perceptron ryml)

add_executable(tester tester.cpp)
target_include_directories(tester PRIVATE thirdparty/rapidyaml/src)
target_link_libraries(tester ryml)
