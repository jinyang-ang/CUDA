/* Start Header *****************************************************************/

/*! \file torus.tese

    \author Ang Jin Yang, jinyang.ang 2000940

    \par email : jinyang.ang.digipen.edu

    \date 1/4/2024

    \brief Copyright (C) 2024 DigiPen Institute of Technology.

Reproduction or disclosure of this file or its contents without
the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/
#version 450

layout (binding = 1) uniform UBO 
{
	mat4 projection;
	mat4 modelview;
	float outerRadius; 
	float tubeRadius;
	vec3 center;
} ubo; 


layout(quads, equal_spacing, cw) in;
layout (location = 0) out vec3 colour;

#define PI 3.14159265359 

vec4 getTorusPosition(float u, float v) {
    float phi = clamp(u * 2.0 * PI,0.0,2*PI);
    float theta = clamp(v * 2.0 * PI,0.0,2*PI);

    float x = (ubo.outerRadius + ubo.tubeRadius * cos(theta)) * cos(phi);
    float y = (ubo.outerRadius + ubo.tubeRadius * cos(theta)) * sin(phi);
    float z = ubo.tubeRadius * sin(theta);

    return vec4(x, y, z,1.f);
}


void main()
{
	gl_Position = getTorusPosition(gl_TessCoord.x,gl_TessCoord.y);
	colour = (gl_Position.x * vec3(1.0,0.0,0.0)) + (gl_Position.y * vec3(0.0,1.0,0.0)) + (gl_Position.z * vec3(0.0,0.0,1.0)); 
	gl_Position = ubo.projection * ubo.modelview * (gl_Position + vec4(ubo.center,1.f));
}