cmake_minimum_required(VERSION 3.20)
project(yamcc)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

include(FetchContent)
FetchContent_Declare(SDL2 GIT_REPOSITORY "https://github.com/libsdl-org/SDL.git" GIT_TAG "release-2.24.1")
FetchContent_Declare(spdlog GIT_REPOSITORY "https://github.com/gabime/spdlog.git" GIT_TAG "v1.10.0")

FetchContent_MakeAvailable(SDL2 spdlog)

file(GLOB YAMCC_SOURCES src/*.c src/*.cpp)
add_executable(yamcc WIN32 ${YAMCC_SOURCES})

target_link_libraries(yamcc SDL2 SDL2main spdlog)
