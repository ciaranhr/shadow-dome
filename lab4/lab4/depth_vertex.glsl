#version 330 

in vec3 positionAttribute;
in vec3	normalAttribute;
in vec2	texCoordAttribute;


uniform mat4 modelToClipTransform;


void main()
{
    gl_Position = modelToClipTransform * vec4(positionAttribute, 1.0);
}
