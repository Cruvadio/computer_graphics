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
    Material plastic(glm::vec3(1.0,1.0,1.0), 0.0, 0.5);
    Material white (glm::vec3(1.0,1.0,1.0));
    Material green(glm::vec3(0.4,1.0,0.4));
    Material rubber (glm::vec3(0.8,0.5,1.0), 0.8, 0.0);
    Material glass (glm::vec3(1,1,1), 0.0, 1.0, 1);
    std::vector<Object*> objects;
    std::vector<Light> lights;
    Sphere sphere1 = Sphere(glm::vec3(-3, 0, -25), 3.0f, plastic);
    Sphere sphere2 = Sphere(glm::vec3(5.0, -1.5, -18), 2.0f,rubber);
    Plane floor(0, 1, 0, 4, 10);
    Plane wall1(-1, 0, 0, 10, 32, white);
    Plane wall2(1, 0, 0, 10, 32, white);
    Plane wall3(0, 0, 1, 32, 10, green);
    Plane potolok(0, -1, 0, 12, 32, white);

    //Sphere sphere3 = Sphere(glm::vec3(-12, 4, -2), 2.0f,plastic);
    objects.push_back(&sphere1);
    objects.push_back(&sphere2);
    objects.push_back(&floor);
    objects.push_back(&wall1);
    objects.push_back(&wall2);
    objects.push_back(&wall3);
    objects.push_back(&potolok);
    lights.push_back(Light(0.8, vec3(10.5), glm::vec3(5,8,-25),glm::vec3(1.0,1.0,1.0)));
    //lights.push_back(Light(0.5, vec3(3.5), glm::vec3( -7, 7, -25),glm::vec3(1.0,1.0,1.0)));
    
    //lights.push_back(Light(0.5,vec3(4.5), glm::vec3(7,0,-14),glm::vec3(1.0,1.0,1.0)));

    render(objects, lights);

    return 0;
}