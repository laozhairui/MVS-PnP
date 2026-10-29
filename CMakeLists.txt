cmake_minimum_required(VERSION 3.12)

project(MVS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(vino_task/src/YoloVino)

set(MVS_INCLUDE_DIR "/opt/MVS/include")
set(MVS_LIBRARY_DIR "/opt/MVS_Runtime/MVS/lib/64")

add_executable(MVS
    src/main.cpp
    src/Camera.cpp
)

target_include_directories(MVS PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${MVS_INCLUDE_DIR}
)

target_link_directories(MVS PRIVATE
    ${MVS_LIBRARY_DIR}
)

target_link_libraries(MVS PRIVATE
    ${OpenCV_LIBS}
    MvCameraControl
    YoloVino_LIB
)
