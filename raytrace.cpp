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

//***************** VECTOR *****************//
class Vector {
    public:
        float x, y, z;
        Vector();
        Vector(float, float, float);
        Vector(point, point);
        Vector add(Vector);
        Vector sub(Vector);
        Vector mul(Vector);
        Vector div(Vector);
        void normalize();

};

//TODO: Actually Implement the things above

//***************** NORMAL *****************//
class Normal {
    float x, y, z;
    Normal(float, float, float);
    Normal add(Normal);
    Normal add(Normal);
};

//TODO: Actually Implement the things above

//***************** POINT *****************//
class Point {
    float x, y, z;
    Point(float, float, float);
    Point plus(Vector);
    Point minus(Vector);
};

//TODO: Actually Implement the things above

//***************** RAY *****************//
class Ray {
    /* Represents the ray:
       r(t) = pos + t*dir*/
    Point pos;
    Vector dir;
    float t_min, t_max;
};

//TODO: Actually Implement the things above

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
    float r, g, b;
    Color(float, float, float);
    Color add(Color);
    Color sub(Color);
    Color mult(float);
    Color dif(float);
};

//TODO: Actually Implement the things above

//***************** CAMERA *****************//
class Camera {
    Point lookfrom;
    Point lookat;
    Vector up;
    float fov;
    Camera();
    Camera(Point, Point, Vector, float);
    Ray generateRay();
};

//TODO: Actually Implement the things above

//****************************************************
// Global Variables
//****************************************************
float imgwidth, imgheight;
int maxdepth = 5;
string filename;
Camera eye;


//****************************************************
// Testing Code
//****************************************************

//****************************************************
// the usual stuff, nothing exciting here
//****************************************************
int main(int argc, char *argv[]) {
  return 0;
}
