#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include <unistd.h>
#include <iostream>
//#include <object.h>
#include "vulkan.hpp"



int main (int argc, char** argv)
{
    
    ComputeApplication app;
    float r1 = ((float)rand() / (RAND_MAX));
    float r2 = ((float)rand() / (RAND_MAX));
    float r3 = ((float)rand() / (RAND_MAX));
    std::cout << "(" << r1 << " " << r2 << " " << r3 << ")\n";
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