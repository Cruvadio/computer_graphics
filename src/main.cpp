#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <ctime>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <iostream>
//#include <object.h>
#include "vulkan.hpp"



int main (int argc, char** argv)
{
    
  #ifdef NDEBUG
  srand(time(NULL));
  #endif
  std::unordered_map<std::string, std::string> cmdLineParams;
  for (int i = 0; i < argc; i++)
  {
    std::string key(argv[i]);

    if (key.size() > 0 && key[0] == '-')
    {
      if (i != argc - 1)
      {
        cmdLineParams[key] = argv[i + 1];
        i++;
      }
      else 
        cmdLineParams[key] = "";
    }
  }

  std::string outFilePath = "out.bmp";
  if (cmdLineParams.find("-out") != cmdLineParams.end())
    outFilePath = cmdLineParams["-out"];

  int sceneId = 1;
  if(cmdLineParams.find("-scene") != cmdLineParams.end())
    sceneId = atoi(cmdLineParams["-scene"].c_str());

  if (sceneId > 1)
  {
    std::cout << "No scene has found.\n";
    return EXIT_SUCCESS;
  }
  int numThreads = 1;
  if(cmdLineParams.find("-threads") != cmdLineParams.end())
    numThreads = atoi(cmdLineParams["-threads"].c_str());

  if (numThreads > 1)
  {
    std::cout << "No multithreads avaliable. Run with 1 thread.\n";
  }

  ComputeApplication app;
  try
  {
    app.run(outFilePath.c_str());
  }
  catch (const std::runtime_error& e)
  {
    printf("%s\n", e.what());
    return EXIT_FAILURE;
  }
    
  return EXIT_SUCCESS;
}