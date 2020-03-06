#version 330 core
out vec4 FragColor;
  
uniform vec4 objectColor;
uniform vec4 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;

uniform sampler2D ourTexture;

in vec3 FragPos;  
in vec3 Normal;

void main()
{
    float ambientStrength = 0.1;
    float specularStrength = 0.5;

    vec3 ambient = vec3(ambientStrength * lightColor.x, ambientStrength * lightColor.y, ambientStrength * lightColor.z);

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos); 

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm); 

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
    vec3 specular = specularStrength * spec * vec3(lightColor);  

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(lightColor);

    FragColor = vec4(ambient + diffuse + specular, 1.0f) * objectColor;
}