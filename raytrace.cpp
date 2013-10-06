
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef OSX
#include <GLUT/glut.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/glu.h>
#endif

#include <time.h>
#include <math.h>


#define PI 3.14159265  // Should be used from mathlib
inline float sqr(float x) { return x*x; }

using namespace std;

//****************************************************
// Some Classes
//****************************************************

class Viewport;
class Rgb;
class Light;
class Vec3;

class Viewport {
  public:
    int w, h; // width and height
};

class Rgb {
  public:
    float red, green, blue;
    Rgb();
    Rgb(float, float, float);
};

Rgb::Rgb() {
  red = 0.0f;
  green = 0.0f;
  blue = 0.0f;
}

Rgb::Rgb(float r, float g, float b) {
    red = r;
    green = g;
    blue = b;
}

class Light {
  public:
    float x, y, z;
    Rgb rgb;
    bool isPL;
    Light();
    Light(float, float, float, Rgb, bool);
};

Light::Light() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
  rgb = Rgb ();
  isPL = false;
}

Light::Light(float a, float b, float c, Rgb color, bool PL) {
  x = a;
  y = b;
  z = c;
  rgb = color;
  isPL = PL;
}

class Vec3 {
  public: 
    float x, y, z;
    Vec3();
    Vec3(float, float, float);
    Vec3 dirToLight(Light);
    float dot(Vec3);
    Vec3 times(float);
    Vec3 sub(Vec3);

};

Vec3::Vec3() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
}

Vec3::Vec3(float a, float b, float c) {
  // constructs a vector, normalizes it if it is not normalized.
    float len = sqrt(a*a + b*b + c*c);
    x = a/len;
    y = b/len;
    z = c/len;
}

Vec3 Vec3::dirToLight(Light l) {
  float a = l.x - x;
  float b = l.y - y;
  float c = l.z - z;

  float len = sqrt(pow(a,2) + pow(b,2) + pow(c,2));
  a = a/len;
  b = b/len;
  c = c/len;

  return Vec3(a,b,c);
}

float Vec3::dot(Vec3 v) {
  return x*v.x + y*v.y + z*v.z;
}

Vec3 Vec3::times(float k) {
  float a = k*x;
  float b = k*y;
  float c = k*z;
  return Vec3(a,b,c);
}

Vec3 Vec3::sub(Vec3 v) {
  float a = x - v.x;
  float b = y - v.y;
  float c = z = v.z;

  float len = sqrt(pow(a,2) + pow(b,2) + pow(c,2));
  a = a/len;
  b = b/len;
  c = c/len;
  return Vec3(a,b,c);
}


//****************************************************
// Global Variables
//****************************************************


//****************************************************
// Simple init function
//****************************************************

}


//****************************************************
// reshape viewport if the window is resized
//****************************************************

//****************************************************


//****************************************************
// the usual stuff, nothing exciting here
//****************************************************
int main(int argc, char *argv[]) {
  return 0;
}








