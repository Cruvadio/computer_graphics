#include <stdio.h>
#include <stdlib.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <string>
#include <unistd.h>
#include <object.h>



int main (int argc, char** argv)
{
    Material plastic(0.0, glm::vec3(1.0,1.0,1.0));
    std::vector<Object*> objects;
    std::vector<Light> lights;
    Sphere sphere = Sphere(glm::vec3(-3.0f, 0.0f, -16.0f), 2.0f,plastic);
    objects.push_back(&sphere);
    lights.push_back(Light(3, glm::vec3(-20,20,20),glm::vec3(1.0,1.0,1.0)));

    render(objects, lights);

    return 0;
}