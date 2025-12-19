cmake_minimum_required(VERSION 3.16.3)
project(cmake_test)

find_package(MPI REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 20)
   
add_executable(Code code/cpp/main.cpp)

target_link_libraries(Code PUBLIC MPI::MPI_CXX)

set(KAGEN_USE_SPARSEHASH OFF)

add_subdirectory(extern/KaGen)
target_link_libraries(Code PUBLIC KaGen::KaGen)

add_subdirectory(extern/kamping)
target_link_libraries(Code PUBLIC kamping::kamping)

set(MESSAGE_QUEUE_USE_BOOST ON)

add_subdirectory(extern/message-queue)
target_link_libraries(Code PUBLIC message-queue::message-queue)

add_subdirectory(extern/fmt)
target_link_libraries(Code PUBLIC fmt::fmt)

target_compile_options(Code PUBLIC -O3 -march=native)
