cmake_minimum_required(VERSION 3.9)
project(HMatrix)

set(CMAKE_CXX_STANDARD 11)

add_executable(HMatrix main.cpp cluster.h cluster.cpp QuadTree.h BinaryTree.h)