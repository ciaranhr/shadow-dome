#version 330 core
in vec2 TexCoords;
in vec4 FragPosLightSpace;

out vec4 FragColor;

uniform sampler2D texture1;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

float ShadowCalculation(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = 0.005;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}

void main()
{
    vec3 color = texture(texture1, TexCoords).rgb;
    vec3 ambient = 0.3 * color;

    vec3 lightColor = vec3(1.0);
    vec3 lightDir = normalize(lightPos - FragPosLightSpace.xyz);
    float diff = max(dot(lightDir, vec3(0, 0, 1)), 0.0);
    vec3 diffuse = diff * lightColor;

    float shadow = ShadowCalculation(FragPosLightSpace);
    vec3 lighting = (ambient + (1.0 - shadow) * diffuse) * color;
    
    FragColor = vec4(lighting, 1.0);
}