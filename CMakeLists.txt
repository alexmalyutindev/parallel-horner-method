cmake_minimum_required(VERSION 3.13)
project(Sonay10)

set(CMAKE_C_COMPILER /usr/local/Cellar/open-mpi/4.0.0/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/local/Cellar/open-mpi/4.0.0/bin/mpic++)

add_executable(Sonay10 parallel.cpp)