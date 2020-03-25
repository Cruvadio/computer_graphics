cmake_minimum_required(VERSION 3.8)
project(path_tracer)
add_subdirectory (external)

find_package(Vulkan)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

include_directories(
	include/
	external/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-5.0.1/include/
	external/assimp-5.0.1/build/code
	external/assimp-5.0.1/build/include
	external/RandLib-v.1.0/
	external/random-master/include
	.
)
set (CMAKE_CXX_STANDARD 11)

include_directories(${Vulkan_INCLUDE_DIR})

set(ALL_LIBS
	${OPENGL_LIBRARY}
	${Vulkan_LIBRARY}
	glfw
	GLEW_1130
	effolkronium_random
)

add_executable(path_tracer src/main.cpp src/Bitmap.cpp)

set_target_properties(path_tracer PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

#set(ASSIMP_LIBRARY_DIRS "external/assimp-5.0.1/build/code")
#link_directories(${ASSIMP_LIBRARY_DIRS})




target_link_libraries(path_tracer
	${ALL_LIBS}
)


file(COPY shaders/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)