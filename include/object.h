#ifndef OBJECT_H_
#define OBJECT_H_

#include <glm/glm.hpp>
#include <omp.h>
//#include <openacc.h>
#include<glm/common.hpp>
#include<cstdlib>
#include<ctime>
#include <vector>
#include <fstream>
#include<limits>
#include<iostream>
#include "effolkronium/random.hpp"

#define WIDTH 1024
#define HEIGHT 768

#define SAMPLES 128

#define MAX_DEPTH 16
#define MAX_DISTANCE 1000

#define PI 3.14159265358979f


#define BLACK vec3(0.0f, 0.0f, 0.0f)
using namespace glm;
using Random = effolkronium::random_static;

struct Material
{
    float roughness;
    float metallic;
    vec3 albedo;
    float a0;
    vec3 F0;
    bool isMirror;

    Material(float roughness = 0.0, const vec3& albedo = vec3(1, 0, 0), float metallic = 0.0, float a0 = 1.0, bool isMirror = false) : 
                roughness (roughness),
                metallic(metallic),
                albedo(albedo),
                a0(a0),
                isMirror(isMirror)
                {
                    if (isMirror) F0 = vec3(1.0f);
                    else
                    {
                        F0 = vec3(0.04, 0.04, 0.04);
                        F0 = mix(F0, albedo, metallic);
                    }
                }



    



    vec3 CookTorranceBRDF (const vec3& N,const vec3& V,const vec3& L)
    {   
        if (isMirror) return vec3(1.0);

        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        float ambientStrength = 0.2;
        float specularStrength = 0.8;

        //vec3 ambient = vec3(ambientStrength * lightColor.x, ambientStrength * lightColor.y, ambientStrength * lightColor.z);

        vec3 reflectDir = reflect(-L, N); 

        float spec = pow(max(dot(V, reflectDir), 0.0f), 32);
        vec3 specular = specularStrength * spec * vec3(lightColor);  

        float diff = max(dot(N, L), 0.0f);
        vec3 diffuse = diff * vec3(lightColor);

        return(diffuse + specular) * albedo;
        /*
        vec3 F = fresnelSchlick(std::max(dot(H, V), 0.0f));
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);

        vec3 numerator = NDF * F * G;
        float denominator = 4.0 * std::max(dot(N, V), 0.0f) * std::max(dot(N, L), 0.0f) + 1e-3f;
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;

        kD = kD * (1.0f - metallic);

        return kD * albedo / PI + specular;*/
    }

    vec3 fresnelSchlick(float cosTheta)
    {
        if (isMirror) return F0;
        return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
    }
    
private:

    

    float DistributionGGX(const vec3& N, const vec3& H, float roughness)
    {
        float a      = roughness*roughness;
        float a2     = a*a;
        float NdotH  = std::max(dot(N, H), 0.0f);
        float NdotH2 = NdotH*NdotH;
	
        float num   = a2;
        float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
        denom = PI * denom * denom;
	
        return num / denom;
    }

    float GeometrySchlickGGX(float NdotV, float roughness)
    {
        float r = (roughness + 1.0f);
        float k = (r*r) / 8.0f;

        float num   = NdotV;
        float denom = NdotV * (1.0f - k) + k;
	
        return num / denom;
    }
    float GeometrySmith(const vec3& N, const vec3& V, const vec3& L, float roughness)
    {
        float NdotV = std::max(dot(N, V), 0.0f);
        float NdotL = std::max(dot(N, L), 0.0f);
        float ggx2  = GeometrySchlickGGX(NdotV, roughness);
        float ggx1  = GeometrySchlickGGX(NdotL, roughness);
	
        return ggx1 * ggx2;
    } 

    vec3 mix (const vec3& x, const vec3& y, float a)
    {
        return x * (1 - a) + y * a;
    }


};



struct Object
{
    Material material;

    Object(const Material& mat): material(mat) {}
    virtual bool intersect (const vec3& ray_origin, const vec3& ray_direction, float &t0) const = 0;
    virtual vec3 normal (const vec3& hit) const = 0;
};



struct Sphere : Object
{
    vec3 center;
    float radius;
    Sphere (const vec3& center, float radius, const Material& mat = Material()): 
            Object(mat),
            center(center),
            radius(radius)
            {}
    virtual bool intersect (const vec3& ray_origin, const vec3& ray_direction, float &t0) const
    {
        vec3 L = center - ray_origin;
        float tca = dot(L, ray_direction);
        float d2 = dot(L, L) - dot(tca, tca);
        if (d2 > radius * radius) return false;
        float thc = sqrt(radius*radius - d2);
        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }

    virtual vec3 normal (const vec3& hit) const
    {
        return normalize(hit - center);
    }

};

struct Light
{
    float radius;
    float intensity;
    vec3 lightPosition;
    vec3 lightColor;


    Light(float radius,float intensity, const vec3& lightPos, const vec3& lightCol):
            radius(radius),
            intensity(intensity),
            lightPosition(lightPos),
            lightColor(lightCol)
            {}

    bool intersect (const vec3& ray_origin, const vec3& ray_direction, float &t0) const
    {
        return Sphere(lightPosition, radius).intersect(ray_origin, ray_direction, t0);
    }

    vec3 normal (const vec3& hit) const
    {
        return normalize(hit - lightPosition);
    }

    float surfaceArea() const
    {
        return 4 * PI * radius * radius;
    }
};
/*
struct Box : Object
{
    vec3 min;
    vec3 max;

    Box (const vec3& min, const vec3& max,const Material& mat = Material()):
            Object(mat),
            min(min),
            max(max)
            {}

    virtual bool intersect (const vec3& ray_origin, const vec3& ray_direction, float &t0) const
    {
        vec3 inv_dir = vec3(1.0f/ray_direction.x,1.0f/ray_direction.y, 1.0f/ray_direction.z);
        float lo = inv_dir.x * (min.x - ray_origin.x);
        float hi = inv_dir.x * (max.x - ray_origin.x);
        float tMin = min(lo, hi);
        float tMax = max(lo, hi);

        float lo1 = inv_dir.y * (min.y - ray_origin.y);
        float hi1 = inv_dir.y * (max.y - ray_origin.y);
        tMin = min(tMin, min(lo1, hi1));
        tMax = max(tMax, max(lo1, hi1));

        float lo2 = inv_dir.z * (min.z - ray_origin.z);
        float hi2 = inv_dir.z * (max.z - ray_origin.z);
        tMin = max(tMin, min(lo2, hi2));
        tMax = min(tMax, max(lo2, hi2));

        t0 = tMin;
        return (tMin <= tMax) && (tMin > 0.f)
    }

};*/

struct Plane: Object
{
    float A;
    float B;
    float C;
    float D;

    //float size;

    Plane(float A, float B, float C, float D = 0, float size = 0,const Material &mat = Material()) :
                Object(mat),
                A (A),
                B (B),
                C (C),
                D (D)
             //   size (size)
                {}

    

    virtual bool intersect(const vec3& ray_origin, const vec3& ray_direction, float &t0) const
    {
        float L = dot(ray_direction, normal());
        if ( L <= 1e-3f  && L >= -1e-3f) return false;
        t0 = - (dot(ray_origin, normal()) + D)/L; 
        return t0 > 0;
    }

    
    virtual vec3 normal (const vec3& hit = vec3()) const
    {
        return normalize(vec3(A, B, C));
    }
};

/*
struct Triangle: Object
{
    vec3 v0;
    vec3 v1;
    vec3 v2;

    //float size;

    Triangle(vec3 v0, vec3 v1, vec3 v2, Material mat = Material()) :
                Object(mat),
                v0(v0),
                v1(v1),
                v2(v2)
             //   size (size)
                {}

    

    virtual bool intersect(const vec3& ray_origin, const vec3& ray_direction, float &t0) const
    {
        float L = dot(ray_direction, normal());
        if (L == 0) return false;
        t0 = - (dot(ray_origin, normal()) + ray_direction)/L; 
        return true;
    }

    
    virtual vec3 normal (const vec3& hit = vec3()) const
    {
        return normalize(cross(v1 - v0, v2 - v0));
    }
};
*/

vec3 random_unit_vector_in_hemisphere_of(const vec3& normal)
{
    float x = (float) Random::get(-1.0f, 1.0f);
    float y = (float) Random::get(-1.0f, 1.0f);
    float z = (float) Random::get(-1.0f, 1.0f);
    
    if (dot(normal, vec3(x,y,z)) < 0)
        return normalize(-vec3(x,y,z));
    return normalize(vec3(x,y,z));
}

bool scene_intersect(const vec3 &ray_origin, const vec3& ray_direction, const std::vector<Object*>& objects, const std::vector<Light> lights, vec3& hit, vec3& N, Material &material, bool& isLight, vec3& lightColor)
{
    float object_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < objects.size(); i++)
    {
        float dist_i;
        if (objects[i]->intersect(ray_origin, ray_direction, dist_i) && dist_i < object_dist)
        {
            object_dist = dist_i;
            hit = ray_origin + ray_direction * dist_i;
            N = objects[i]->normal(hit);
            material = objects[i]->material;
            isLight = false;
        }
    }

    for (size_t i = 0; i < lights.size(); i++)
    {
        float dist_i;
        if (lights[i].intersect(ray_origin, ray_direction, dist_i) && dist_i < object_dist)
        {
            object_dist = dist_i;
            hit = ray_origin + ray_direction * dist_i;
            N = lights[i].normal(hit);
            isLight = true;
            lightColor = lights[i].lightColor;
        }
    }
    
    return object_dist < MAX_DISTANCE;
}



vec3 cast_ray (const vec3& ray_orig, const vec3 &ray_dir, const std::vector<Object*>& objects, const std::vector<Light>& lights, size_t depth = 0, const vec3& prevHit_pos = vec3(), const vec3& prevHit_norm = vec3())
{
    vec3 point, N;
    Material material;
    bool isLight = false;
    vec3 lightColor;

    if (depth > MAX_DEPTH || !scene_intersect(ray_orig, ray_dir, objects, lights, point, N, material, isLight, lightColor))
    {
        return BLACK; // background color
    }

    if (isLight)
    {
        float D = distance(prevHit_pos, point);
        float pE = D * D/ (dot(N, -ray_dir));
        float pI = dot(prevHit_norm, ray_dir)/ PI;
        float wI = pI*pI/(pE*pE + pI*pI);
        return lightColor /** wI*/;
    }

    vec3 L0 = vec3(0.0);
    vec3 BDRF = vec3();
    vec3 new_ray_dir = !material.isMirror? random_unit_vector_in_hemisphere_of(N) : normalize(reflect(ray_dir, N));
    float cosTheta = dot(new_ray_dir, N);

    for (size_t i = 0; i < lights.size(); i++)
    {
        vec3 L = lights[i].lightPosition - point;
        vec3 H = normalize(ray_dir + L);
        vec3 an_point;
        vec3 an_N;
        Material an_mat;

        float distance    = length(lights[i].lightPosition - point);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance     = lights[i].lightColor * attenuation * lights[i].intensity;   

        
        bool shadow = scene_intersect(point + vec3(1e-4), normalize(L), objects, lights, an_point, an_N, an_mat, isLight, lightColor) && !isLight;
        float kSh = 1.0;
        if (shadow)
        {
            kSh = 1e-3;
            continue;
        }

        //std::cout<< "not shadow\n";
        vec3 sdir = normalize(L);
        float cosTheta1 =fabs( dot(sdir, an_N));
        float cosTheta2 = fabs(dot(sdir, N));
        
        //float distance = length(L);
        float lgtPdf =  distance * distance/ cosTheta1;
        vec3 lgtVal = lights[i].intensity * (material.CookTorranceBRDF(N, ray_dir,normalize(L)) * cosTheta2/PI);

        float NdotL = max(dot(N, L), 0.0f);  
  
        //L0 += kSh * material.CookTorranceBRDF(N, ray_dir,normalize(L)) * radiance * NdotL;

        float pI = cosTheta/PI;
        float wE = lgtPdf * lgtPdf / (lgtPdf * lgtPdf + pI * pI);
        L0 += kSh*wE *lgtVal/lgtPdf;
    }

    //vec3 ambient = vec3(0.05) * material.albedo * material.a0;
    //vec3 color = ambient + L0;
    
    // Gamma-correction
    L0 = L0 / (L0 + vec3(1.0));
    L0 = pow(L0, vec3(1.0/1.6)); 

    //std::cout<< "(" << color.x << ", " << color.y << ", " << color.z << ")\n";
    
    /*
    vec3 reflect_dir = normalize(reflect(ray_dir, N));
    vec3 refract_dir = normalize(refract(ray_dir, N, material.refractive_index));
    vec3 reflect_orig = dot(reflect_dir, N) < 0 ? point - N * 1e-3 : point + N * 1e-3;
    vec3 refract_orig = dot(refract_dir, N) < 0 ? point - N * 1e-3 : point + N * 1e-3;
    vec3 reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, depth + 1);
    vec3 refract_color = cast_ray(refract_orig, refract_dir, spheres, depth + 1);*/
    

    return  (!material.isMirror ? L0 : vec3(0.0)) + material.CookTorranceBRDF(N, ray_dir, new_ray_dir) * cast_ray(point, new_ray_dir, objects, lights, depth + 1, point, N);    
}


//vec3 light_trace ()


void render (const std::vector<Object*>& objects, const std::vector<Light> &lights)
{
    const float fov = PI / 2.;
    std::vector<vec3> framebuffer(WIDTH * HEIGHT);
    omp_set_num_threads(8);
    #pragma omp parallel for
   /// #pragma acc kernels loop independent
    for (size_t j = 0; j < HEIGHT; j++)
        for (size_t i = 0; i < WIDTH; i++)
        {
            float dir_x = (i + 0.5) - (float) WIDTH/2.;
            float dir_y = -(j + 0.5) + (float) HEIGHT/2.;
            float dir_z = -(float)WIDTH/ (tan(fov/2.));
            vec3 direction = normalize(vec3(dir_x, dir_y, dir_z));
            framebuffer[i + j * WIDTH] = vec3(0.0);
            //std::cout<< "(" << direction.x << ", " << direction.y << ", " << framebuffer[i + j * WIDTH].z << ")\n";
            for (int k = 0; k < SAMPLES; k++)
            {
                srand(time(NULL));
                framebuffer[i + j* WIDTH] += cast_ray (vec3(0.0), direction, objects, lights);
            }
            vec3 color =  framebuffer[i + j * WIDTH] / (float) SAMPLES;
            //color = color / (color + vec3(1.0));
            //color = pow(color, vec3(1.0/2.2)); 
            framebuffer[i + j * WIDTH] = color;
           // std::cout<< "(" << framebuffer[i + j * WIDTH].x << ", " << framebuffer[i + j * WIDTH].y << ", " << framebuffer[i + j * WIDTH].z << ")\n";
        }

    std::ofstream ofs;
    ofs.open("./out.ppm", std::ios::binary);
    ofs << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (size_t i = 0; i < HEIGHT * WIDTH; ++i)
    {
        vec3 &c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max > 1) c = c * (1.f/max);
        for (size_t j = 0; j < 3; j++)
        {
            ofs << (char) (255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }        
    }
}

#endif