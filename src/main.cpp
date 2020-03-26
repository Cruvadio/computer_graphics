#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <ctime>
#include <string>
#include <unistd.h>
#include <iostream>
//#include <object.h>
#include "vulkan.hpp"



int main (int argc, char** argv)
{
    
  #ifdef NDEBUG
  srand(time(NULL));
  #endif
    ComputeApplication app;
  try
  {
    app.run("out.bmp");
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS;
}