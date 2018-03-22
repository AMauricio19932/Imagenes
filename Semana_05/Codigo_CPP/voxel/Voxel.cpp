#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
#include <vector>
#include <math.h>
#include <cfloat>
#include "GL/glew.h"
#include "GL/glut.h"
#include "Angel-yjc.h"
#include "stb_image.c"
using namespace std;

#define WIDTH  600
#define HEIGHT 600
#define PI 3.14159265

GLuint Angel::InitShader(const char* vShaderFile, const char* fShaderFile);
void reducTri(int p1, int p2, int p3);

GLuint cube_buffer, grid_buffer;

static GLuint texName, textureID;

mat4 rot = Rotate(0, 1, 1, 1);


vec3 verticesRay[8]={
  vec3(-0.5f,-0.5f,-0.5f),
	vec3( 0.5f,-0.5f,-0.5f),
	vec3( 0.5f, 0.5f,-0.5f),
	vec3(-0.5f, 0.5f,-0.5f),

	vec3(-0.5f,-0.5f, 0.5f),
	vec3( 0.5f,-0.5f, 0.5f),
	vec3( 0.5f, 0.5f, 0.5f),
	vec3(-0.5f, 0.5f, 0.5f)
};

GLushort cubeIndicesRay[36]={
  0,5,4,
  5,0,1,
  3,7,6,
  3,6,2,
  7,4,6,
  6,4,5,
  2,1,3,
  3,1,0,
  3,0,7,
  7,0,4,
  6,5,2,
  2,5,1
};


typedef Angel::vec4  color4;
typedef Angel::vec3  point3;
typedef Angel::vec4  vec4;


static int value = 0;

std::vector<point3> gridMap;
std::vector<point3> gridMapNormal;
std::vector<point3> gridMapColor;
std::vector<vec2> gridMapTexCoordr;
std::vector<int> indObjetos;

std::vector<int> indMap;
std::vector<int> indMapNormal;
std::vector<int> indMapTexCoordr;


int numNodes, widthTex, heightTex, componentsTex;
unsigned char *texColor;

GLuint program;
GLuint program2;
bool flag_gird, firstMouse = 1;
float yaw, pitch;
vec4 cameraFront;

GLfloat  fovy = 20.0;
GLfloat  aspect;
GLfloat  zNear = .2, zFar = 5.0;

vec4 VRP = vec4(-3.0, 0.0, 0.0, 1.0);
vec4 VPN = vec4(1.0, 0.0, 0.0, 0.0);
vec4 VUP = vec4(0.0, 1.0, 0.0, 0.0);
vec4 eye = vec4(0.0, 0.0, 0.0, 0.0);

float R = 0.01, mx, my, mz, Mx, My, Mz;


void readTri(string a){
  std::vector<int> v = {0, 0, 0};
  char c;
  int j = 0, i;
  for (i = 0; i < a.size(); i++) {
    c = a[i];
    if (c == '/') {
      j++;
    } else {
      v[j] *= 10;
      v[j] += (c - '0');
    }
  }
  indMap.push_back(v[0] - 1);

}
point3 pointM;
void readFace(string s){
  mx = FLT_MAX; my = FLT_MAX; my = FLT_MAX;
  Mx = -FLT_MAX; My = -FLT_MAX; My = -FLT_MAX;
  freopen(s.c_str(),"r", stdin);
  float x, y, z;
  string t;
  cin>> t;
  while (t == "v") {
    cin>> x >> y >> z;
    x /= 100; y /= 100; z /= 100;
    gridMap.push_back(point3( x, y, z));
    if(mx > x) mx = x;
    if(my > y) my = y;
    if(mz > z) mz = z;

    if(Mx < x) Mx = x;
    if(My < y) My = y;
    if(Mz < z) Mz = z;
    cin>> x >> y >> z;
    gridMapColor.push_back(point3( x, y, z));
    gridMapNormal.push_back(point3( 0, 0, 0));
    indObjetos.push_back(0);

    cin>> t;
  }
  while (t == "vt") {
    cin>> x >> y;
    gridMapTexCoordr.push_back(vec2( x, y));
    cin>> t;
  }
  while (t == "vn") {
    cin>> x >> y >> z;
    gridMapNormal.push_back(point3( x, y, z));
    cin>> t;
  }
  while (t != "f") {
    cin>> t;
  }

  pointM = point3(((Mx - mx)/2) + mx,((My - my)/2) + my,((Mz - mz)/2) + mz );
  while (t == "f") {
    for (size_t i = 0; i < 3; i++) {
      cin>> t;
      readTri(t);
    }

    cin>> t;

  }
  numNodes = indMap.size();
  std::cout << numNodes << '\n';
  std::cout << mx << ' ' << my << ' ' << mz << ' ' << '\n';
  std::cout << Mx << ' ' << My << ' ' << Mz << ' ' << '\n';
  std::cout << pointM << '\n';

}

float distacia(int p1, int p2){
  float ac;
  ac = pow(gridMap[p1].x - gridMap[p2].x,2);
  ac += pow(gridMap[p1].y - gridMap[p2].y,2);
  ac += pow(gridMap[p1].z - gridMap[p2].z,2);
  return sqrt(ac);
}

void addPoint (int p1, int p2, int p3){
  float x, y, z;
  x = (gridMap[p1].x + gridMap[p2].x)/2;
  y = (gridMap[p1].y + gridMap[p2].y)/2;
  z = (gridMap[p1].z + gridMap[p2].z)/2;
  int p4 = gridMap.size();
  gridMap.push_back(point3( x, y, z));
  reducTri(p1, p4, p3);
  reducTri(p2, p4, p3);
}

void reducTri(int p1, int p2, int p3){
  float l[3], lm = 0;
  int i, j;
  l[0] = distacia(p1, p2);
  l[1] = distacia(p2, p3);
  l[2] = distacia(p3, p1);
  for (i = 0; i < 3; i++) {
    if (lm < l[i]) {
      lm = l[i];
      j = i;
    }
  }
  if(lm > R){
    if (j == 0) addPoint(p1, p2, p3);
    else if (j == 1) addPoint(p2, p3, p1);
    else addPoint(p3, p1, p2);
  }
}

int  minNum(std::vector<int> v){
  int m = v[0];
  for (size_t j = 1; j < v.size(); j++) {
    if (m > v[j]) m = v[j];
  }
  return m;
}

int  maxNumInd(std::vector<int> v){
  int m = v[0], i = 0;
  for (size_t j = 1; j < v.size(); j++) {
    if (m < v[j]) {
      m = v[j];
      i = j;
    }
  }
  return i;
}

void objetos(){
  int g = 1, ind, tem, M;
  bool flagObj;
  std::vector<int> relaVec, indVec, endObj, tamaObj;
  relaVec.push_back(0);
  for (size_t i = 0; i < indMap.size(); i += 3) {
    indVec.clear();
    for (size_t j = i; j < i + 3; j++) {
      tem = indObjetos[indMap[j]];
      while (tem > 0) {
        indVec.push_back(tem);
        tem = relaVec[tem];
      }
    }
    if (!indVec.empty()) {
      ind = minNum(indVec);
      for (size_t j = 0; j < indVec.size(); j++) {
        if (indVec[j] != ind) {

          relaVec[indVec[j]] = ind;
        }
      }
      indObjetos[indMap[i]] = ind; indObjetos[indMap[i + 1]] = ind; indObjetos[indMap[i + 2]] = ind;
    }
    else {
      indObjetos[indMap[i]] = g; indObjetos[indMap[i + 1]] = g; indObjetos[indMap[i + 2]] = g;
      relaVec.push_back(0);
      g++;
    }
  }
  for (size_t i = 1; i < relaVec.size(); i++) {
    tem = i;
    int c = 0;
    while (relaVec[tem] > 0) {
      tem = relaVec[tem];
      c++;
    }
    if (relaVec[i] > 0) {
      relaVec[i] = tem;
    }
    flagObj = 1;
    for (size_t j = 0; j < endObj.size(); j++) {
      if (endObj[j] == tem) {
        flagObj = 0;
        break;
      }
    }
    if (flagObj) {
      endObj.push_back(tem);
      tamaObj.push_back(0);
    }
  }
  for (size_t i = 0; i < indObjetos.size(); i++) {
    tem = relaVec[indObjetos[i]];
    if (tem > 0) {
      indObjetos[i] = tem;
    }
    tem = indObjetos[i];
    for (size_t j = 0; j < endObj.size(); j++) {
      if (endObj[j] == tem) {
        tamaObj[j]++;
        break;
      }
    }
  }
  M = maxNumInd(tamaObj);
  M = endObj[M];
  int i = 0;
  while (i < indMap.size()) {

    if (indObjetos[indMap[i]] != M) {
      indMap.erase (indMap.begin()+i);
      indMap.erase (indMap.begin()+i);
      indMap.erase (indMap.begin()+i);
    }else{
      i += 3;
    }
  }

  for (size_t i = 0; i < indObjetos.size(); i++) {

    if (indObjetos[i] != M) {

      gridMap[i] = point3(0.0, 0.0, 0.0);
    }
  }
  std::cout << "fin obj" << '\n';
  for (size_t i = 0; i < endObj.size(); i++) {

    std::cout << endObj[i] << ": "<<tamaObj [i]<<'\n';
  }
}

int logCube = 128;
GLubyte pData[128*128*128];
int XDIM = logCube, YDIM = logCube, ZDIM =logCube;

void newPoints(){
  long long int lx, ly, lz, i, x, y, z, p;
  lx = (long long int )((Mx - mx)/R) + 5;
  ly = (long long int )((My - my)/R) + 5;
  lz = (long long int )((Mz - mz)/R) + 5;

  int* voxInd = (int*) calloc(lx*ly*lz,sizeof(int));
  float* voxX = (float*) calloc(lx*ly*lz,sizeof(float));
  float* voxY = (float*) calloc(lx*ly*lz,sizeof(float));
  float* voxZ = (float*) calloc(lx*ly*lz,sizeof(float));
  for (i = 0; i < indMap.size(); i += 3) {
    reducTri(indMap[i], indMap[i + 1], indMap[i + 2]);
  }
  printf("falla 2, %lld - %lld - %lld \n", lx, ly, lz);
  for (i = 0; i < gridMap.size(); i++) {
    x = (long long int )((gridMap[i].x - mx)/R) + 2;
    y = (long long int )((gridMap[i].y - my)/R) + 2;
    z = (long long int )((gridMap[i].z - mz)/R) + 2;
    p = x + (y*lx) + (z*lx*ly);
    voxInd[p]++;

    voxX[p] += gridMap[i].x;
    voxY[p] += gridMap[i].y;
    voxZ[p] += gridMap[i].z;
  }
  gridMap.clear();


  for (x = 2; x < lx - 2; x++)
    for (y = 2; y < ly - 2; y++){
      for (z = 2; z < lz - 2; z++) {
        p = x + (y*lx) + (z*lx*ly);
        if (voxInd[p] > 0){
          gridMap.push_back(point3(voxX[p]/voxInd[p], voxY[p]/voxInd[p], voxZ[p]/voxInd[p]));
          //gridMap.push_back(point3((x - 1.5)*R + mx, (y - 1.5)*R + my, (z - 1.5)*R + mz));
          gridMapColor.push_back(point3(1.0,0.0,0.0));
          voxInd[p] = 3;
          pData[x + (y*logCube) + (z*logCube*logCube)] = (GLubyte)200;
        }
      }
    }

}



void init(){

  readFace("./media/last_05.obj");
  objetos();
  newPoints();


  float dx = 0.5f / XDIM;
  float dy = 0.5f / YDIM;
  float dz = 0.5f / ZDIM;

  float x0 = 0.0f + dx;
  float x1 = 1.0f - dx;
  float y0 = 0.0f + dy;
  float y1 = 1.0f - dy;
  float z0 = 0.0f + dz;
  float z1 = 1.0f - dz;
  vec3 texc[] =
  {
    vec3( x0,y0,z0 ), vec3( x1,y0,z0 ), vec3( x1,y1,z0 ), vec3( x0,y1,z0 ),
    vec3( x0,y0,z1 ), vec3( x1,y0,z1 ), vec3( x1,y1,z1 ), vec3( x0,y1,z1 ),
  };

  glActiveTexture(GL_TEXTURE0);


	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_3D, textureID);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 4);

	glTexImage3D(GL_TEXTURE_3D,0,GL_RED,XDIM,YDIM,ZDIM,0,GL_RED,GL_UNSIGNED_BYTE,pData);

	glGenerateMipmap(GL_TEXTURE_3D);

  glGenBuffers(1, &cube_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, cube_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(point3)*36*2, NULL, GL_STATIC_DRAW);

  unsigned int sizeData = 0;

  for (int i = 0; i < 36; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), verticesRay[cubeIndicesRay[i]]);
    sizeData += sizeof(point3);
  }
  for (int i = 0; i < 36; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), texc[cubeIndicesRay[i]]);
    sizeData += sizeof(point3);
  }

  program = InitShader("vshader3.glsl", "fshader3.glsl");
  glEnable( GL_DEPTH_TEST );
  glClearColor( 0.529, 0.807, 0.92, 0.0 );
  glLineWidth(2.0);
}

void drawCube(GLuint buffer, int num_vertices)
{
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GLuint vPosition = glGetAttribLocation(program, "vPosition");
  glEnableVertexAttribArray(vPosition);
  glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );

  GLuint vTexCoord = glGetAttribLocation(program, "vTexCoord");
  glEnableVertexAttribArray(vTexCoord);
  glVertexAttribPointer(vTexCoord, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(sizeof(point3) * num_vertices) );

  glDrawArrays(GL_TRIANGLES, 0, num_vertices);

  glDisableVertexAttribArray(vPosition);
  glDisableVertexAttribArray(vTexCoord);
}

void display( void ){

  GLuint  mv_matrix;
  GLuint  proj_matrix;
  GLuint  view_matrix;
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glUseProgram(program);
  mv_matrix   = glGetUniformLocation(program, "MVP");
  view_matrix = glGetUniformLocation(program, "view_matrix");
  proj_matrix  = glGetUniformLocation(program, "proj_matrix");
  glUniform1i( glGetUniformLocation(program, "volume"), 0 );

  mat4  p = Perspective(fovy, aspect, zNear, zFar);
  glUniformMatrix4fv(proj_matrix, 1, GL_TRUE, p);
  mat4 m = Translate(eye) * rot;
  mat4  v = LookAt(VRP, VPN, VUP);
  vec3 camPos = (inverse(upperLeftMat3(rot)))*vec3(VRP[0] - eye[0],VRP[1] - eye[1],VRP[2] - eye[2]);

  glUniformMatrix4fv(mv_matrix, 1, GL_TRUE, p*v*m);
  glUniformMatrix4fv(view_matrix, 1, GL_TRUE, v);
  glUniform3f(glGetUniformLocation(program, "camPos"), camPos[0], camPos[1], camPos[2]);
  glUniform3f(glGetUniformLocation(program, "step_size"), 1.0f/XDIM, 1.0f/YDIM, 1.0f/ZDIM);

  if(flag_gird)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


  drawCube(cube_buffer, 36);

  glutSwapBuffers();
}

void reshape(int width, int height)
{
    glViewport(0, 0, width, height);
    aspect = (GLfloat) width  / (GLfloat) height;
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
  float a = 0.120;
  float angle = (10 / PI);
	switch (key) {
	case 033:
	case 'q': case 'Q':
		exit(EXIT_SUCCESS);
		break;

	case GLUT_KEY_UP: eye[0] += 1.0; break;
	case 'w': eye[0] += a; break;
	case 's': eye[0] -= a; break;
	case 'a': eye[2] += a; break;
	case 'd': eye[2] -= a; break;
	case 'Z': eye[1] -= a; break;
	case 'z': eye[1] += a; break;
	case '4': rot = Rotate(angle, 0, 1, 0)*rot;break;
	case '6': rot = Rotate(angle, 0, -1, 0)*rot;break;
	case '8': rot = Rotate(angle, 0, 0, 1)*rot;break;
	case '2': rot = Rotate(angle, 0, 0, -1)*rot;break;
  case 'g': flag_gird = 1 - flag_gird; break;

	case ' ':
		eye = vec4(0.0, 0.0, 0.0, 0.0);
    rot = Rotate(0, 1, 1, 1);
		break;
	}
	glutPostRedisplay();

}

int main(int argc, char **argv) {
  int err;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutCreateWindow("Practica04");
  err = glewInit();
  if (GLEW_OK != err)
  { printf("Error: glewInit failed: %s\n", (char*) glewGetErrorString(err));
    exit(1);
  }

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);

  init();
  glutMainLoop();
  return 0;
}
