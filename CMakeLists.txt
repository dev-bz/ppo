cmake_minimum_required(VERSION 3.14)
file(GLOB my_SRCS *.cpp)
add_subdirectory(ui)
add_subdirectory(ui/pd)
add_subdirectory(caffe_mods)
add_subdirectory(env)
add_subdirectory(util)

set(CMAKE_SHARED_LIBRARY_PREFIX)
set(CMAKE_SHARED_LIBRARY_SUFFIX)
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_SHARED_LINKER_FLAGS  -Wl,--no-undefined)

link_directories(AFTER ../thrid/lib64)
# message(WARNING ${my_SRCS})
add_library(a.out SHARED
					${my_SRCS})
target_link_libraries(a.out -lEGL -lGLESv3 -lnative_app_glue -landroid -llog -lbox2d -lcaffeproto -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lopenblas -lprotobuf -lglog -lgflags)
# add_link_options(-lEGL -lGLESv3)

target_include_directories(a.out PUBLIC ../thrid/include . util)
set_source_files_properties(main.cpp
	PROPERTIES INC hello)
# set(CMAKE_CXX_FLAGS -std=c++11)
message(WARNING $ENV{TMPDIR})

add_definitions(-DCPU_ONLY)

install(TARGETS a.out DESTINATION $ENV{TMPDIR})