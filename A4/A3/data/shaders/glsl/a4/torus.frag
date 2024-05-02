/* Start Header *****************************************************************/

/*! \file torus.frag

    \author Ang Jin Yang, jinyang.ang 2000940

    \par email : jinyang.ang.digipen.edu

    \date 1/4/2024

    \brief Copyright (C) 2024 DigiPen Institute of Technology.

Reproduction or disclosure of this file or its contents without
the prior written consent of DigiPen Institute of Technology is prohibited. */

/* End Header *******************************************************************/
#version 450
layout (location = 0) in vec3 colour;

layout (location = 0) out vec4 outFragColor;

void main()
{
	outFragColor = vec4(colour,1.0);
}