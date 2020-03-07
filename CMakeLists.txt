cmake_minimum_required(VERSION 3.8)

add_subdirectory (external)

include_directories(
	include/
	external/
	external/glfw-3.1.2/include/
	external/glm-0.9.7.1/
	external/glew-1.13.0/include/
	external/assimp-5.0.1/include/
	external/assimp-5.0.1/build/code
	external/assimp-5.0.1/build/include
	.
)

set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
	assimp
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)
set(ASSIMP_LIBRARY_DIRS "external/assimp-5.0.1/build/code")
link_directories(${ASSIMP_LIBRARY_DIRS})

add_executable(main src/main1.cpp)


target_link_libraries(main
	${ALL_LIBS}
)
