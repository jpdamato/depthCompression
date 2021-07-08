CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project(depth_compression)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/include/gdal)
include_directories(./rapidjson/include)


message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(  ${OpenCV_INCLUDE_DIRS} )

set(OpenCV_DIR "/usr/lib/opencv")


find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui ml)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


File(GLOB SRC_FILES src/*.cpp )

message(src: ${SRC_FILES} )

add_executable(depth_compression main.cpp ${SRC_FILES} )

target_link_libraries( depth_compression ${OpenCV_LIBS} ${GDAL_LIBRARIES} stdc++fs )

SET(GCC_COVERAGE_LINK_FLAGS    "-fPIE"  )