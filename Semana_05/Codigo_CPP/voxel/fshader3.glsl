#version 120

in vec3 fPosition;
in vec3 fTexCoord;


varying out vec4 fColor;

uniform sampler3D volume;
uniform float trfunc_delta = 0.01;
uniform vec3 camPos;
uniform float sample_step = 1.0 / 80.0;
uniform float val_threshold = 0.5;
uniform float isoValue = 40/255.0;

vec4 PhongLighting(vec3 L, vec3 N, vec3 V, float specPower, vec3 diffuseColor)
{
	float diffuse = max(dot(L,N),0.0);
	vec3 halfVec = normalize(L+V);
	float specular = pow(max(0.00001,dot(halfVec,N)),specPower);
	return vec4((diffuse*diffuseColor + specular),1.0);
}

vec3 GetGradient(vec3 uvw)
{
	vec3 s1, s2;

	s1.x = texture3D(volume, uvw-vec3(trfunc_delta,0.0,0.0)).x ;
	s2.x = texture3D(volume, uvw+vec3(trfunc_delta,0.0,0.0)).x ;

	s1.y = texture3D(volume, uvw-vec3(0.0,trfunc_delta,0.0)).x ;
	s2.y = texture3D(volume, uvw+vec3(0.0,trfunc_delta,0.0)).x ;

	s1.z = texture3D(volume, uvw-vec3(0.0,0.0,trfunc_delta)).x ;
	s2.z = texture3D(volume, uvw+vec3(0.0,0.0,trfunc_delta)).x ;

	return normalize((s1-s2)/2.0);
}

vec3 Bisection(vec3 left, vec3 right , float iso)
{
	for(int i=0;i<4;i++)
	{
		vec3 midpoint = (right + left) * 0.5;
		float cM = texture3D(volume, midpoint).x ;
		if(cM < iso)
			left = midpoint;
		else
			right = midpoint;
	}
	return vec3(right + left) * 0.5;
}

void main() {
  const float brightness = 10.0;

  vec3 geomDir = -normalize(camPos - fPosition);
  vec3 ray_dir = geomDir * sample_step;
  vec3 ray_pos = fTexCoord.xyz;
  vec3 pos111 = vec3(1.0, 1.0, 1.0);
  vec3 pos000 = vec3(0.0, 0.0, 0.0);

  vec4 color = vec4(0.529, 0.807, 0.92, 1.0);
  bool stop = false;
  do
  {
      ray_pos += ray_dir ;

      stop = dot(sign(ray_pos-pos000),sign(pos111-ray_pos)) < 3.0;

  		if (stop)
  			break;

      float density = texture3D(volume, ray_pos).r;
  		float density2 = texture3D(volume, ray_pos + ray_dir).r;


      if( (density - isoValue) < 0  && (density2 - isoValue) >= 0.0)  {

  			vec3 xN = ray_pos;
  			vec3 xF = ray_pos + ray_dir;
  			vec3 tc = Bisection(xN, xF, isoValue);

  			vec3 N = GetGradient(tc);
  			vec3 V = geomDir;
  			vec3 L =  -V;

  			color =  PhongLighting(L,N,V,50, vec3(0.5, 0.5, 0.5));
				break;
  		}

  }
  while(true);

  fColor = color;

}
