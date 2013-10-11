#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

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
#include "FreeImage.h"


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
class LocalGeo;

//***************** POINT *****************//
class Point {
  public:
    float x, y, z;
    Point();
    Point(float, float, float);
    Point plus(Vector);
    Point minus(Vector);
};

//***************** VECTOR *****************//
class Vector {
  public:
    float x, y, z, len;
    Vector();
    Vector(float, float, float);
    Vector(Point, Point);
    Vector add(Vector);
    Vector sub(Vector);
    Vector mult(float);
    Vector div(float);
    float dot(Vector);
    Vector cross(Vector);
    void normalize();
    bool equals(Vector);
};

//***************** POINT METHODS *****************//
Point::Point() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
}

Point::Point(float a, float b, float c) {
  x = a;
  y = b;
  z = c;
}

Point Point::plus(Vector v) {
  float a = x + v.x;
  float b = y + v.y;
  float c = z + v.z;
  return Point(a, b, c);
}

Point Point::minus(Vector v) {
  float a = x - v.x;
  float b = y - v.y;
  float c = z - v.z;
  return Point(a, b, c);
}


//***************** VECTOR METHODS *****************//

Vector::Vector() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
  len = 0.0f;
}

Vector::Vector(float a, float b, float c) {
  x = a;
  y = b;
  z = c;
  len = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
}

Vector::Vector(Point start, Point end) {
  x = end.x - start.x;
  y = end.y - start.y;
  z = end.z - start.z;
  len = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
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
  float l = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
  if(len != 0) {
    x = x/l;
    y = y/l;
    z = z/l;
    len = 1.0f;
  }
}

bool Vector::equals(Vector v) {
  return (x == v.x) && (y == v.y) && ( z == v.z);
}

//***************** NORMAL *****************//
class Normal {
  public:
    float x, y, z;
    Normal();
    Normal(float, float, float);
    Normal add(Normal);
    Normal sub(Normal);
    bool equals(Normal);
};

Normal::Normal() {
    x, y, z == 0.0;
}

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

bool Normal::equals(Normal n) {
    return (x == n.x) && (y == n.y) && (z == n.z);
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
  Matrix m, minvt;
  //TODO: should support transformations by overloading *
};

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
    //Ray generateRay();
};
Camera::Camera() {
    lookfrom = Point();
    lookat = Point();
    up = Vector();
    fov = 0.0;
}

Camera::Camera(Point from, Point at, Vector v, float f) {
    lookfrom = from;
    lookat = at;
    up = v;
    fov = f;
}
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
  public:
    Point a, b, c;
};

//***************** LOCALGEO *****************//
class LocalGeo {
  public:
    Point pos;
    Normal n;
    LocalGeo(Point, Normal);
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
bool testVector(string* error) {
  Vector a = Vector(0.5, 0.5, 0.5);
  Vector b = Vector(0.2, 0.2, 0.2);
  Vector c = Vector(0.7, 0.7, 0.7);
  Vector d = Vector(0.3, 0.3, 0.3);
  Vector e = Vector(1.0, 1.0, 1.0);
  Vector f = Vector(1.0, 1.0, 1.0);

  Vector aplusb = a.add(b);
  Vector aminusb = a.sub(b);
  Vector atimes2 = a.mult(2.0);
  Vector ediv2 = e.div(2.0);
  f.normalize();

  if (!aplusb.equals(c)) {
    *error = "Vector addition failed";
    return false;
  } else if (!aminusb.equals(d)) {
    *error = "Vector subtraction failed";
    return false;
  } else if (!atimes2.equals(e)) {
    *error = "multiplicaiton failed";
    return false;
  } else if (!ediv2.equals(a)) {
    *error = "division failed";
    return false;
  } else if(a.dot(e) != 1.5) {
    *error = "dot product failed";
    return false;
  } else if(f.len != 1.0f) {
    *error = "normalize failed";
    return false;
  } else {
    *error = "passed tests";
    return true;
  }
}

bool testNormal(string* error) {
    Normal a = Normal(1.0, 1.0, 1.0);
    Normal b = Normal(2.0, 2.0, 2.0);

    Normal c = Normal(1.0, 0.0, 0.0);
    Normal d = Normal(2.0, 0.0, 0.0);
    Normal e = Normal(-1.0, 0.0, 0.0);
    Normal zero = Normal();

    Normal csubd = c.sub(d);
    Normal eaddd = e.add(d);
    printf("eaddd x: %.20f, y: %.20f, z: %.20f", eaddd.x, eaddd.y, eaddd.z);

    if(!a.equals(b)) {
      *error = "normal equality failed";
      return false;
    } else if(!eaddd.equals(zero)) {
      *error = "normal addition failed";
      return false;
    } else if(!csubd.equals(zero)) {
      *error = "normal subraction failed";
      return false;
    } else {
      *error = "passed tests";
      return true;
    }
}

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
        maxdepth = atoi(splitline[1].c_str());
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
        eye.lookfrom = Point(atof(splitline[1].c_str()), atof(splitline[2].c_str()), atof(splitline[3].c_str()));
          // lookat:
          eye.lookat = Point(atof(splitline[4].c_str()), atof(splitline[5].c_str()), atof(splitline[6].c_str()));
          // up:
          eye.up = Vector(atof(splitline[7].c_str()), atof(splitline[8].c_str()), atof(splitline[9].c_str()));
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
    }
  }
}
//****************************************************
// the usual stuff, nothing exciting here
//****************************************************
int main(int argc, char *argv[]) {
  string error = "no error";
  bool vectest = testVector(&error);
  printf("vectest returned with message: %s \n", error.c_str());
  bool normaltest = testNormal(&error);
  //printf("normaltest returned with message: %s \n", error.c_str());
  FreeImage_Initialise();
  cout << "FreeImage " << FreeImage_GetVersion() << "\n";
  cout << FreeImage_GetCopyrightMessage() << "\n\n";
  FIBITMAP * bitmap = FreeImage_Allocate(100, 100, 24);
  FreeImage_Save(FIF_PNG, bitmap, "testimage.png", 0);
  FreeImage_DeInitialise();
}
