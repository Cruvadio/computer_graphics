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
	external/RandLib-v.1.0/
	external/random-master/include
	.
)
SET(MYCOMPILE_FLAGS "-g -fopenmp -O2")
set(ALL_LIBS
	${OPENGL_LIBRARY}
	glfw
	GLEW_1130
	effolkronium_random
)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${MYCOMPILE_FLAGS}")
add_definitions(
	-fopenmp
)
#set(ASSIMP_LIBRARY_DIRS "external/assimp-5.0.1/build/code")
#link_directories(${ASSIMP_LIBRARY_DIRS})

add_executable(main src/main.cpp)


target_link_libraries(main
	${ALL_LIBS}
)
