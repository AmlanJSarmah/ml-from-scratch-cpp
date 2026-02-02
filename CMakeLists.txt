cmake_minimum_required(VERSION 3.20)
project(MLFromScratch
    VERSION 0.1.0
    DESCRIPTION "C++ from-scratch ML algorithms"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

add_library(ml STATIC
    src/linear_regression.cpp
    src/utils.cpp
    src/logistic_regression.cpp
)

target_include_directories(ml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src
)

target_link_libraries(ml
    PUBLIC
        Eigen3::Eigen
)

add_executable(linear_regression apps/linear_regression/main.cpp)
add_executable(logistic_regression apps/logistic_regression/main.cpp)
target_link_libraries(linear_regression PRIVATE ml)
target_link_libraries(logistic_regression PRIVATE ml)
