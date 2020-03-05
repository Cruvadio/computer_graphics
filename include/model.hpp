#ifndef _MODEL_HPP
#define _MODEL_HPP

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

using namespace std;

struct Vertex
{
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

struct Texture 
{
    unsigned int id;
    string type;
}; 

class Mesh 
{
    public:
        /*  Mesh Data  */

        vector<Vertex> vertices;
        vector<unsigned int> indices;
        vector<Texture> textures;

        /*  Functions  */


        Mesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<Texture> textures);
        void Draw(Shader shader);
    private:
        /*  Render data  */

        unsigned int VAO, VBO, EBO;
        /*  Functions    */

        void setupMesh();
};

#endif