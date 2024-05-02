/* Start Header *****************************************************************/

/*! \file torus.tesc

    \author Ang Jin Yang, jinyang.ang 2000940

    \par email : jinyang.ang.digipen.edu

    \date 1/4/2024

    \brief Copyright (C) 2024 DigiPen Institute of Technology.

Reproduction or disclosure of this file or its contents without
the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/
#version 450

layout (binding = 0) uniform UBO 
{
	float tessLevelO;
	float tessLevelI;

} ubo; 
 
layout (vertices = 4) out;
 
void main()
{
	if (gl_InvocationID == 0)
	{
		gl_TessLevelInner[0] = ubo.tessLevelI;
		gl_TessLevelInner[1] = ubo.tessLevelI;

		gl_TessLevelOuter[0] = ubo.tessLevelO;
		gl_TessLevelOuter[1] = ubo.tessLevelO;
		gl_TessLevelOuter[2] = ubo.tessLevelO;	
		gl_TessLevelOuter[3] = ubo.tessLevelO;		
	
	}

	gl_out[gl_InvocationID].gl_Position =  gl_in[gl_InvocationID].gl_Position;
} 
