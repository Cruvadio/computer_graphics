#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <main.hpp>

using namespace glm;


int main (int argc, char** argv)
{
  if (!glfwInit())
  {
    fprintf(stderr, "Error while initializing glfw");
    return 1;
  }

  glfwWindowHint(GLFW_SAMPLES, 4); // 4x Сглаживание
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // Мы хотим использовать OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Мы не хотим старый OpenGL

  // Открыть окно и создать в нем контекст OpenGL
  GLFWwindow* window; // (В сопроводительном исходном коде эта переменная является глобальной)
  window = glfwCreateWindow( WIDTH, HEIGHT, "Task 01", NULL, NULL);
  if( window == NULL ){
	   fprintf( stderr, "Невозможно открыть окно GLFW. Если у вас Intel GPU, то он не поддерживает версию 3.3. Попробуйте версию уроков для OpenGL 2.1.n" );
	    glfwTerminate();
	     return -1;
     }
  glfwMakeContextCurrent(window);

     // Инициализируем GLEW
  glewExperimental=true; // Флаг необходим в Core-режиме OpenGL
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Невозможно инициализировать GLEWn");
    return -1;
  }
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

  do{
    // Пока что ничего не выводим. Это будет в уроке 2.

    // Сбрасываем буферы
      glfwSwapBuffers(window);
      glfwPollEvents();

    } // Проверяем нажатие клавиши Escape или закрытие окна
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
  glfwWindowShouldClose(window) == 0 );

  return 0;
  
}
