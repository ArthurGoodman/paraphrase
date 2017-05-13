cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "${CMAKE_X_FLAGS} -Wall")

project(paraphrase)

file(GLOB_RECURSE sources "src/*.cpp")
file(GLOB_RECURSE headers "src/*.h")

# include_directories(".")

set(EXECUTABLE_OUTPUT_PATH "bin")

add_executable(paraphrase ${sources} ${headers})
