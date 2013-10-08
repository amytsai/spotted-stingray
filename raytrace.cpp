#include <iostream>
#include <fstream>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef OSX
#else
#endif

#include <time.h>
#include <math.h>


#define PI 3.14159265  // Should be used from mathlib
inline float sqr(float x) { return x*x; }

using namespace std;

//****************************************************
// Some Classes
//****************************************************
class Vector;
class Normal;
class Point;
class Ray;
class Matrix;
class Transformation;
class Color;
class Camera;
class Shape;
class Sphere;
class Triangle;

//***************** VECTOR *****************//
class Vector {
  public:
    float x, y, z;
    Vector();
    Vector(float, float, float);
    Vector(point, point);
    Vector add(Vector);
    Vector sub(Vector);
    Vector mult(float);
    Vector div(float);
    float dot(Vector);
    Vector cross(Vector);
    void normalize();

};

Vector::Vector() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
}

Vector::Vector(float a, float b, float c) {
  x = a;
  y = b;
  z = c;
}

Vector::Vector(point start, point end) {
  x = end.x - start.x;
  y = end.y - start.y;
  z = end.z - start.z;
}

Vector Vector::add(Vector v) {
  float a = x + v.x;
  float b = y + v.y;
  float c = z + v.z;

  return Vector(a,b,c);
}

Vector Vector::sub(Vector v) {
  float a = x - v.x;
  float b = y - v.y;
  float c = z - v.z;

  return Vector(a,b,c);
}

Vector Vector::mult(float k) {
  float a = k*x;
  float b = k*y;
  float c = k*z;

  return Vector(a,b,c);
}

Vector Vector::div(float k) {
  float a = x/k;
  float b = y/k;
  float c = z/k;

  return Vector(a,b,c);
}

float Vector::dot(Vector v) {
  return x*v.x + y*v.y + z*v.z;
}

Vector Vector::cross(Vector v) {
  float a = y * v.z - z * v.y;
  float b = z * v.x - x * v.z;
  float c = x * v.y - y * v.x;

  return Vector(a, b, c);
}

void Vector::normalize() {
  float len = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
  if(len != 0) {
	  x = x/len;
	  y = y/len;
	  z = z/len;
  }
}

//***************** NORMAL *****************//
class Normal {
    public:
      float x, y, z;
      Normal(float, float, float);
      Normal add(Normal);
      Normal sub(Normal);
};

Normal::Normal(float a, float b, float c) {
  x = a;
  y = b;
  z = c;
  if(x != 0 || y !=0 || z !=0) {
    float len = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    x = x/len;
    y = y/len;
    z = z/len;
  }
}

Normal Normal::add(Normal v) {
  float a = x + v.x;
  float b = y + v.y;
  float c = z + v.z;

  return Normal(a,b,c);
}

Normal Normal::sub(Normal v) {
  float a = x - v.x;
  float b = y - v.y;
  float c = z - v.z;

  return Normal(a,b,c);
}

//***************** POINT *****************//
class Point {
  public:
    float x, y, z;
    Point();
    Point(float, float, float);
    Point plus(Vector);
    Point minus(Vector);
};

Point::Point() {
  x = 0.0f;
  y = 0.0f;
  y = 0.0f;
}

Point::Point(float a, float b, float c) {
  x = a;
  y = b;
  z = c;
}

Point Point::plus(Vector v) {
  a = x + v.x;
  b = y + v.y;
  c = z + v.z;
  return Point(a, b, c);
}

Point Point::minus(Vector v) {
  a = x - v.x;
  b = y - v.y;
  c = z - v.z;
  return Point(a, b, c);
}

//***************** RAY *****************//
class Ray {
  /* Represents the ray:
     r(t) = pos + t*dir*/
  public:
    Point pos;
    Vector dir;
    float t_min, t_max;
    Ray(Point, Point);
    Ray(Point, Vector);
};

Ray::Ray(Point a, Point b) {
  pos = a;
  dir = Vector(a, b);
}

Ray::Ray(Point a, Vector v) {
  pos = a;
  dir = v;
}


//***************** MATRIX *****************//
class Matrix {
  float mat[4][4];
  //TODO: Figure out what a matrix should be able to do
};


//***************** TRANSFORMATION *****************//
class Transformation {
  Matrix m, minvt
    //TODO: should support transformations by overloading *
}

//***************** COLOR *****************//
class Color {
  public:
    float r, g, b;
    Color();
    Color(float, float, float);
    Color add(Color);
    Color sub(Color);
    Color mult(float);
    Color div(float);
};

Color::Color() {
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;
}

Color::Color(float a, float b, float c) {
  r = a;
  g = b;
  b = c;
}

Color Color::add(Color v) {
  float a = r + v.r;
  float b = g + v.g;
  float c = b + v.b;

  return Color(a,b,c);
}

Color Color::sub(Color v) {
  float a = r - v.r;
  float b = g - v.g;
  float c = b - v.b;

  return Color(a,b,c);
}

Color Color::mult(float k) {
  float a = k*r;
  float b = k*g;
  float c = k*b;

  return Color(a,b,c);
}

Color Color::div(float k) {
  float a = r/k;
  float b = g/k;
  float c = b/k;

  return Color(a,b,c);
}

//***************** CAMERA *****************//
class Camera {
  public:
    Point lookfrom;
    Point lookat;
    Vector up;
    float fov;
    Camera();
    Camera(Point, Point, Vector, float);
    Ray generateRay();
};

//TODO: Actually Implement the things above

//***************** SHAPE *****************//
class Shape {
    virtual bool intersect(Ray& ray, float* thit, LocalGeo* local);
    virtual bool ifIntersect(Ray& ray);
};
//***************** SPHERE *****************//
class Sphere: public Shape {
  public:
    Point pos;
    float r;
    Sphere(Point, float);
    bool intersect(Ray& ray, float* thit, LocalGeo* local);
    bool ifIntersect(Ray& ray);

};
//***************** TRIANGLE *****************//
class Triangle : public Shape {

};
//****************************************************
// Global Variables
//****************************************************
float width, height;
int maxdepth = 5;
string filename;
Camera eye;


//****************************************************
// Testing Code
//****************************************************

//****************************************************
// Parse Scene File
//****************************************************
void loadScene(std::string file) {

  ifstream inpfile(file.c_str());
  if(!inpfile.is_open()) {
    std::cout << "Unable to open file" << std::endl;
  } else {
    std::string line;
    //MatrixStack mst;

    while(inpfile.good()) {
      vector<string> splitline;
      string buf;

      getline(inpfile,line);
      stringstream ss(line);

      while (ss >> buf) {
        splitline.push_back(buf);
      }
      //Ignore blank lines
      if(splitline.size() == 0) {
        continue;
      }

      //Ignore comments
      if(splitline[0][0] == '#') {
        continue;
      }

      //Valid commands:
      //size width height
      //  must be first command of file, controls image size
      else if(!splitline[0].compare("size")) {
        width = atoi(splitline[1].c_str());
        height = atoi(splitline[2].c_str());
      }
      //maxdepth depth
      //  max # of bounces for ray (default 5)
      else if(!splitline[0].compare("maxdepth")) {
        maxdepth = atoi(splitline[1].c_str())
      }
      //output filename
      //  output file to write image to 
      else if(!splitline[0].compare("output")) {
        filename = splitline[1];
      }

      //camera lookfromx lookfromy lookfromz lookatx lookaty lookatz upx upy upz fov
      //  speciï¬es the camera in the standard way, as in homework 2.
      else if(!splitline[0].compare("camera")) {
        // lookfrom:
            eye.lookfrom = Point(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()))
        // lookat:
            eye.lookat = Point(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()))
        // up:
            eye.up = Vector(atof(splitline[7].c_str()), atof(splitline[8].c_str()), atof(splitline[9].c_str()))
        // fov: atof(splitline[10].c_str());
            eye.fov = atof(splitline[10].c_str());
      }

      //sphere x y z radius
      //  Deï¬nes a sphere with a given position and radius.
      else if(!splitline[0].compare("sphere")) {
        // x: atof(splitline[1].c_str())
        // y: atof(splitline[1].c_str())
        // z: atof(splitline[1].c_str())
        // r: atof(splitline[4].c_str())
        // Create new sphere:
        //   Store 4 numbers
        //   Store current property values
        //   Store current top of matrix stack
      }
      //****************************************************
      // the usual stuff, nothing exciting here
      //****************************************************
      int main(int argc, char *argv[]) {
        return 0;
      }
