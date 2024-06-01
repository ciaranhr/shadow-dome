#version 330

// Input from the vertex shader, will contain the interpolated (i.e., distance weighted average) vaule out put for each of the three vertex shaders that 
// produced the vertex data for the triangle this fragment is part of.
// Grouping the 'in' variables in this way makes OpenGL check that they match the vertex shader
in VertexData
{
	vec3 v2f_viewSpaceNormal;
	vec3 v2f_viewSpacePosition;
	vec2 v2f_texCoord;
	vec4 fragmentPosLightSpace;
};

// Material properties set by OBJModel.
uniform vec3 material_diffuse_color; 
uniform float material_alpha;
uniform vec3 material_specular_color; 
uniform vec3 material_emissive_color; 
uniform float material_specular_exponent;

// Textures set by OBJModel names must be bound to the right texture unit, ObjModel.setDefaultUniformBindings helps with that.
uniform sampler2D diffuse_texture;
uniform sampler2D specular_texture;

// Other uniforms used by the shader
uniform vec3 viewSpaceLightPosition;
uniform vec3 lightColourAndIntensity;
uniform vec3 ambientLightColourAndIntensity;
uniform sampler2D shadowMap;

out vec4 fragmentColor;


float ShadowCalculation(vec4 fragPosLightSpace)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float bias = 0.005;
    float shadow = currentDepth > closestDepth + bias ? 1.0 : 0.0;
    return shadow;

	//PCF
	//float shadow = 0.0;
	//vec2 texelsize = 1.0/textureSize(shadowMap, 0);
	//for (int x = -1; x<=1; ++x) {
	//	for (int y = -1; y <= 1; ++y) {
	//		float pcfDepth = texture(shadowMap, projCoords.xy, + vec2(x,y) * texelsize).r;
	//		shadow += d2 - bias > pcfDepth ? 1.0 : 0.0;
	//	}
	//}
	//shadow /= 9.0;
	//if (projCoords.z > 1.0) {
	//	shadow = 0.0;
	//}
	//return shadow
}

void main() 
{
	vec3 materialDiffuse = texture(diffuse_texture, v2f_texCoord).xyz * material_diffuse_color;
	vec3 viewSpaceDirToLight = normalize(viewSpaceLightPosition - v2f_viewSpacePosition);
	vec3 viewSpaceNormal = normalize(v2f_viewSpaceNormal);
	
	float shadow = ShadowCalculation(fragmentPosLightSpace);
	float incomingIntensity = max(0.0, dot(viewSpaceNormal, viewSpaceDirToLight));
	vec3 incomingLight = ((1.0 - shadow)+ incomingIntensity) * lightColourAndIntensity;
	//(1.0 - shadow)
	fragmentColor = vec4(vec3(incomingLight), material_alpha);
}
