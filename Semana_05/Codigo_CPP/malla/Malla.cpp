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

GLuint cube_buffer, grid_buffer;

static GLuint texName;
mat4 rotIni = Rotate(90, 0, 1, 0)*Rotate(180, 1, 0, 0);
mat4 rot = rotIni;

std::vector<vec3> cube_vertices = {
  vec3(-1.0,  1.0,  1.0),
  vec3(-1.0, -1.0,  1.0),
  vec3( 1.0, -1.0,  1.0),
  vec3( 1.0,  1.0,  1.0),
  vec3(-1.0,  1.0, -1.0),
  vec3(-1.0, -1.0, -1.0),
  vec3( 1.0, -1.0, -1.0),
  vec3( 1.0,  1.0, -1.0)
};

GLushort cube_indices[] = {
  0, 1, 2, 3,
  3, 2, 6, 7,
  7, 6, 5, 4,
  4, 5, 1, 0,
  0, 3, 7, 4,
  1, 2, 6, 5,
};

typedef Angel::vec4  color4;
typedef Angel::vec3  point3;
typedef Angel::vec4  vec4;


static int value = 0;

std::vector<point3> gridMap;
std::vector<point3> gridMapColor;
std::vector<point3> gridMapNormal;
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

GLfloat  fovy = 80.0;
GLfloat  aspect;
GLfloat  zNear = .10, zFar = 100.0;

vec4 VRP = vec4(-1.0, 0.0, 0.0, 0.0);
vec4 VPN = vec4(2.0, 0.0, 0.0, 0.0);
vec4 VUP = vec4(0.0, 1.0, 0.0, 0.0);
vec4 eye = vec4(0.0, 0.0, 0.0, 0.0);


float mx, my, mz, Mx, My, Mz;
vec3 pointM = vec3(0.0, 0.0, 0.0);

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

void readFace(string s){
  mx = FLT_MAX; my = FLT_MAX; my = FLT_MAX;
  Mx = -FLT_MAX; My = -FLT_MAX; My = -FLT_MAX;
  freopen(s.c_str(),"r", stdin);
  float x, y, z;
  string t;
  cin>> t;
  while (t == "v") {
    cin>> x >> y >> z;
    gridMap.push_back(point3( x/100, y/100, z/100));
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
  pointM = point3(((Mx - mx)/2) + mx,((My - my)/2) + my,((Mz - mz)/2) + mz )/100;
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
  for (int i = 0; i < gridMap.size(); i++) {
    gridMap[i].x -= pointM.x;
    gridMap[i].y -= pointM.y;
    gridMap[i].z -= pointM.z;
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
    };
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
  std::cout << M << '\n';
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

void init(){
  readFace("./media/last_05.obj");
  objetos();

  glGenBuffers(1, &cube_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, cube_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(point3)*24, NULL, GL_STATIC_DRAW);
  unsigned int sizeData = 0;

  for (int i = 0; i < 24; i++) {

    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), cube_vertices[cube_indices[i]]);
    sizeData += sizeof(point3);
  }

  glGenBuffers(1, &grid_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, grid_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(point3)*numNodes*2, NULL, GL_STATIC_DRAW);

  sizeData = 0;

  for (int i = 0; i < numNodes; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), gridMap[indMap[i]]);
    sizeData += sizeof(point3);
  }

  for (int i = 0; i < numNodes; i++) {
    glBufferSubData(GL_ARRAY_BUFFER, sizeData, sizeof(point3), gridMapColor[indMap[i]]);
    sizeData += sizeof(point3);
  }

  program = InitShader("vshader.glsl", "fshader.glsl");
  glEnable( GL_DEPTH_TEST );
  glClearColor( 0.529, 0.807, 0.92, 0.0 );
  glLineWidth(2.0);
}

void drawObj(GLuint buffer, int num_vertices)
{
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GLuint vPosition = glGetAttribLocation(program, "vPosition");
  glEnableVertexAttribArray(vPosition);
  glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );

  GLuint vColor = glGetAttribLocation(program, "vColor");
  glEnableVertexAttribArray(vColor);
  glVertexAttribPointer(vColor, 3, GL_FLOAT, GL_FALSE, 0,
    BUFFER_OFFSET(sizeof(point3) * num_vertices) );


  glDrawArrays(GL_TRIANGLES, 0, num_vertices);
  glDisableVertexAttribArray(vPosition);
  glDisableVertexAttribArray(vColor);
}

void drawCube(GLuint buffer, int num_vertices)
{
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  GLuint vPosition = glGetAttribLocation(program, "vPosition");
  glEnableVertexAttribArray(vPosition);
  glVertexAttribPointer(vPosition, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0) );

  glDrawArrays(GL_QUADS, 0, num_vertices);

  glDisableVertexAttribArray(vPosition);
}

void display( void ){

  GLuint  mv_matrix;
  GLuint  proj_matrix;
  GLuint  view_matrix;
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
  glUseProgram(program);
  mv_matrix   = glGetUniformLocation(program, "mv_matrix");
  view_matrix = glGetUniformLocation(program, "view_matrix");
  proj_matrix  = glGetUniformLocation(program, "proj_matrix");


  mat4  p = Perspective(fovy, aspect, zNear, zFar);

  glUniformMatrix4fv(proj_matrix, 1, GL_TRUE, p);
  mat4 m = Translate(eye) * rot;

  mat4  v = LookAt(VRP, VPN, VUP);
  vec3 camPos = (inverse(upperLeftMat3(rot)))*vec3(VRP[0] - eye[0],VRP[1] - eye[1],VRP[2] - eye[2]);
  glUniformMatrix4fv(mv_matrix, 1, GL_TRUE, v*m);
  glUniformMatrix4fv(view_matrix, 1, GL_TRUE, v);
  glUniform3f(glGetUniformLocation(program, "eye"), camPos[0], camPos[1], camPos[2]);

  if(flag_gird)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  drawObj(grid_buffer, numNodes);

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
  float a = 0.125;
  float angle = (10 / PI);
	switch (key) {
	case 033:
	case 'q': case 'Q':
		exit(EXIT_SUCCESS);
		break;

	case GLUT_KEY_UP: eye[0] += 1.0; break;
	case 'w': eye[0] += a; break;
	case 's': eye[0] -= a; break;
	case 'a': eye[2] -= a; break;
	case 'd': eye[2] += a; break;
	case 'Z': eye[1] -= a; break;
	case 'z': eye[1] += a; break;
  case '4': rot = Rotate(angle, 0, 1, 0)*rot;break;
  case '6': rot = Rotate(angle, 0, -1, 0)*rot;break;
  case '8': rot = Rotate(angle, 0, 0, 1)*rot;break;
  case '2': rot = Rotate(angle, 0, 0, -1)*rot;break;
  case 'g': flag_gird = 1 - flag_gird; break;
	case ' ':
		eye = vec4(0.0, 0.0, 0.0, 0.0);
    rot = rotIni;
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
