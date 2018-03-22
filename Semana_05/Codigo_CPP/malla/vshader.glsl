
#version 120

in vec3 vPosition;
in vec3 vColor;

varying out vec3 color;
varying out vec3 fPosition;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;


void main()
{
  vec4 vPosition4 = vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);

  color = vColor;
  gl_Position = proj_matrix * mv_matrix * vPosition4;
  fPosition = vPosition;
}
