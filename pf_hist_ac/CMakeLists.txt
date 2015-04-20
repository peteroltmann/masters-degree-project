cmake_minimum_required(VERSION 2.8)

# if the make program is not detected automatically, put the following as cmake
# argument:
# -D"CMAKE_MAKE_PROGRAM:PATH=<absolute path>"
# e.g.:
# -D"CMAKE_MAKE_PROGRAM:PATH=C:/MinGW/bin/make.exe"
#
# Debug:
# -D"CMAKE_BUILD_TYPE=Debug"
# Release:
# -D"CMAKE_BUILD_TYPE=Release"

project(pf_hist_ac)

include_directories(include)
aux_source_directory(src SRC_LIST)

# read out include dir to show up in Qt Creator (add to executable/library)
file(GLOB_RECURSE INCLUDE_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     FOLLOW_SYMLINKS include/*.h *.yml)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

# add INCLUDE_LIST to show up in project tree - won't mess with build process
add_executable(${PROJECT_NAME} ${SRC_LIST} ${INCLUDE_LIST})

if(OpenCV_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

# -DSHOW_CONTOUR_EVOLUTION -DSAVE_AS_VIDEO -DTIME_MEASUREMENT
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
