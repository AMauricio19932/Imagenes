#version 120

in vec3 color;


varying out vec4 fColor;


void main() {
  fColor = vec4(color, 1.0);
}
