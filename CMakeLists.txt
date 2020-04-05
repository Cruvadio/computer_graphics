cmake_minimum_required(VERSION 3.8)
project(path_tracer)
#add_subdirectory (external)

find_package(Vulkan)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

include_directories(
	include/
	external/
	external/glm-0.9.7.1/
	.
)
set (CMAKE_CXX_STANDARD 11)

include_directories(${Vulkan_INCLUDE_DIR})

set(ALL_LIBS
	${Vulkan_LIBRARY}
)

add_executable(path_tracer src/main.cpp src/Bitmap.cpp)

set_target_properties(path_tracer PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

target_link_libraries(path_tracer
	${ALL_LIBS}
)


file(COPY shaders/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)