#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <math.h>

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
#include "FreeImage/FreeImage.h"


#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;


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
class MatrixGenerator;
class Transformation;
class Color;
class Camera;
class Shape;
class Sphere;
class Triangle;
class LocalGeo;

class Sample;
class Sampler;

//****************************************************
//  Global Variables
//****************************************************
int width, height;
int maxdepth = 5;

//***************** POINT *****************//
class Point {
  public:
    Vector4f point;
    Point();
    Point(float, float, float);
	Point(Vector4f);
    Point add(Vector);
    Point minus(Vector);
	Vector minus(Point);
};

//***************** VECTOR *****************//
class Vector {
  public:
    Vector4f vector;
	float len;
    Vector();
    Vector(float, float, float);
    Vector(Point, Point);
	Vector(Vector4f);
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
  Vector4f temp(0, 0, 0, 1);
  point = temp;
}

Point::Point(float a, float b, float c) {
  Vector4f temp(a, b, c, 1);
  point = temp;
}

Point::Point(Vector4f vec) {
	point = vec;
	point(3) = 1;
}

Point Point::add(Vector v) {
  Vector4f temp = point + v.vector;
  return Point(temp);
}

Point Point::minus(Vector v) {
  Vector4f temp = point - v.vector;
  return Point(temp);
}

Vector Point::minus(Point p) {
	Vector4f temp = point - p.point;
	return Vector(temp);
}



//***************** VECTOR METHODS *****************//

Vector::Vector() {
  Vector4f temp(0, 0, 0, 0);
  vector = temp;
  len = vector.norm();
}

Vector::Vector(float a, float b, float c) {
  Vector4f temp(a, b, c, 0);
  vector = temp;
  len = vector.norm();
}

Vector::Vector(Vector4f vec) {
	vector = vec;
	vector(3) = 0;
	len = vector.norm();
}

Vector::Vector(Point start, Point end) {
	vector = end.point - start.point;
	vector(3) = 0;
  len = vector.norm();
}

Vector Vector::add(Vector v) {
  Vector4f temp = vector + v.vector;
  return Vector(temp);
}

Vector Vector::sub(Vector v) {
  Vector4f temp = vector - v.vector;
  return Vector(temp);
}

Vector Vector::mult(float k) {
  Vector4f temp = vector * k;
  return Vector(temp);
}

Vector Vector::div(float k) {
  Vector4f temp = vector / k;
  return Vector(temp);
}

float Vector::dot(Vector v) {
  return vector.dot(v.vector);
}

Vector Vector::cross(Vector v) {
	Vector3f temp1, temp2, temp3;
	temp1 << vector(0), vector(1), vector(2);
	temp2 << v.vector(0), v.vector(1), v.vector(2);
	temp3 = temp1.cross(temp2);
	Vector4f temp4;
	temp4 << temp3(0), temp3(1), temp3(2), 0;
	return Vector(temp4);
}

void Vector::normalize() {
  vector.normalize();
}

bool Vector::equals(Vector v) {
	Vector4f temp = v.vector - vector;
	float size = temp.norm();
  return size == 0;
}

//***************** NORMAL *****************//
class Normal {
  public:
    Vector4f normal;
    Normal();
    Normal(float, float, float);
	Normal(Vector);
	Normal(Vector4f);
    Normal add(Normal);
    Normal sub(Normal);
    bool equals(Normal);
};

Normal::Normal() {
    Vector4f temp(0, 0, 0, 0);
  normal = temp;
}

Normal::Normal(float a, float b, float c) {
  Vector4f temp(a, b, c, 0);
  normal = temp;
  normal.normalize();
}

Normal::Normal(Vector v) {
	normal = v.vector;
	normal.normalize();
}
Normal::Normal(Vector4f vec) {
	normal = vec;
	normal.normalize();
}

Normal Normal::add(Normal v) {
  Vector4f temp = normal + v.normal;
  return Normal(temp);
}

Normal Normal::sub(Normal v) {
  Vector4f temp = normal - v.normal;
  return Normal(temp);
}

bool Normal::equals(Normal n) {
    Vector4f temp = n.normal - normal;
	float size = temp.norm();
  return size == 0;
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
	Point getPoint(float);
};

Ray::Ray(Point a, Point b) {
  pos = a;
  dir = Vector(a, b);
  t_min = 0;
  t_max = 99999999;
}

Ray::Ray(Point a, Vector v) {
  pos = a;
  dir = v;
  t_min = 0;
  t_max = 99999999;
}

Point Ray::getPoint(float time) {
	Vector travel = dir.mult(time);
	Point temp;
	temp = pos.add(travel);
	return temp;
}

//***************** TRANSFORMATION *****************//
class Transformation {
  //Matrix m, minvt;
	public: 
		Matrix4f matrix, matrixinv;
		Transformation(Matrix4f);
  //TODO: should support transformations by overloading *
};

Transformation::Transformation(Matrix4f mat) {
	matrix = mat;
	matrixinv = mat.inverse();
}


//***************** MATRIX *****************//
class MatrixGenerator {
  //float mat[4][4];
  //TODO: Figure out what a matrix should be able to do
	public:
		MatrixGenerator();
		Transformation generateTranslation(float, float, float);
		Transformation generateRotationx(float);
		Transformation generateRotationy(float);
		Transformation generateRotationz(float);
		Transformation generateScale(float, float, float);
};

MatrixGenerator::MatrixGenerator() {
}

Transformation MatrixGenerator::generateTranslation(float x, float y, float z) {
	Matrix4f temp;
	temp << 1, 0, 0, x,
			0, 1, 0, y,
			0, 0, 1, z,
			0, 0, 0, 1;
	return temp;
}

Transformation MatrixGenerator::generateRotationx(float angle) {
	float rad = angle * PI/180;
	Matrix4f temp;
	temp << 1, 0, 0, 0,
			0, cos(rad), -sin(rad), 0,
			0, sin(rad), cos(rad), 0,
			0, 0, 0, 1;
	return temp;
}


Transformation MatrixGenerator::generateRotationy(float angle) {
	float rad = angle * PI/180;
	Matrix4f temp;
	temp << cos(rad), 0, -sin(rad), 0,
			0, 1, 0, 0,
			sin(rad), 0, cos(rad), 0,
			0, 0, 0, 1;
	return temp;
}

Transformation MatrixGenerator::generateRotationz(float angle) {
	float rad = angle * PI/180;
	Matrix4f temp;
	temp << cos(rad), -sin(rad), 0, 0,
			sin(rad), cos(rad), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
	return temp;
}

Transformation MatrixGenerator::generateScale(float x, float y, float z) {
	Matrix4f temp;
	temp << x, 0, 0, 0,
			0, y, 0, 0,
			0, 0, z, 0,
			0, 0, 0, 1;
	return temp;
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
    Vector UL, LL, UR, LR;
    float fov;
    Camera();
    Camera(Point, Point, Vector, float);
    void generateRay(Sample &, Ray* ray);
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
    up.normalize();
    fov = f;

    Vector z = Vector(from, at);
    z.normalize();
    Vector x = z.cross(up);
    x.normalize();
    Vector zminusx = z.sub(x);
    Vector zplusx = z.add(x);

    UL = zminusx.add(up);
    LL = zminusx.sub(up);
    UR = zplusx.add(up);
    LR = zplusx.sub(up);

}

//***************** LOCAL GEO *****************//
class LocalGeo {
  public:
    Point pos;
    Normal n;
    LocalGeo(Point, Normal);
};

LocalGeo::LocalGeo(Point p, Normal norm) {
	pos = p;
	n = norm;
}

//***************** SHAPE *****************//
class Shape {
public:
  virtual bool intersect(Ray&, float*, LocalGeo* ) = 0;
  virtual bool ifIntersect(Ray& ) = 0;
};

//***************** SPHERE *****************//
class Sphere: public Shape {
  public:
    Point pos;
    float r;
    Sphere(Point, float);
    bool intersect(Ray&, float*, LocalGeo*);
    bool ifIntersect(Ray&);
};

Sphere::Sphere(Point p, float rad) {
    pos  = p;
    r = rad;
}

bool Sphere::intersect(Ray& ray, float* thit, LocalGeo* local) {
	Point raystart = ray.pos;
	Vector direction = ray.dir;
	Point center = pos;
	Vector3f e, d, c;
	e << raystart.point(0), raystart.point(1), raystart.point(2);
	d << direction.vector(0), direction.vector(1), direction.vector(2);
	c << center.point(0), center.point(1), center.point(2);
	float determinant = pow(d.dot(e - c), 2) - (d.dot(d))*((e - c).dot(e - c) - r*r);
	if(determinant < 0) {
		return false;
	}
	else {
		float hittime = (sqrt(determinant) + -d.dot(e - c))/(d.dot(d));
		*thit = hittime;
		Point hitPoint = ray.getPoint(hittime);
		Normal norm = Normal((hitPoint.minus(center)));
		*local = LocalGeo(hitPoint, norm);
		return true;
	}

    
}

bool Sphere::ifIntersect(Ray& ray) {
    Point raystart = ray.pos;
	Vector direction = ray.dir;
	Point center = pos;
	Vector3f e, d, c;
	e << raystart.point(0), raystart.point(1), raystart.point(2);
	d << direction.vector(0), direction.vector(1), direction.vector(2);
	c << center.point(0), center.point(1), center.point(2);
	if(pow(d.dot(e - c), 2) - (d.dot(d))*((e - c).dot(e - c) - r*r) < 0) {
		return false;
	}
	else {
		return true;
	}
}

//***************** TRIANGLE *****************//
class Triangle : public Shape {
  public:
   Point a, b, c;
    bool intersect(Ray&, float* , LocalGeo* );
    bool ifIntersect(Ray& );
};

bool Triangle::intersect(Ray& ray, float* thit, LocalGeo* local) {
    return false;
}

bool Triangle::ifIntersect(Ray& ray) {
    return false;
}



//***************** LIGHT *****************//
class Light {
  public:
    float x, y, z;
    Color rgb;
    bool isPL;
    Light();
    Light(float, float, float, Color, bool);
};

Light::Light() {
  x = 0.0f;
  y = 0.0f;
  z = 0.0f;
  rgb = Color ();
  isPL = false;
}

Light::Light(float a, float b, float c, Color color, bool PL) {
  x = a;
  y = b;
  z = c;
  rgb = color;
  isPL = PL;
}

//***************** SAMPLE *****************//
class Sample {
  public:
    //holds screen coordinates;
    float x, y;
    Sample();
};

Sample::Sample() {
    x, y = 0.0;
}
//***************** SAMPLER *****************//
class Sampler {
  public:
    int i, j;
    bool getSample(Sample *);
    Sampler();
};
Sampler::Sampler() {
    i, j = 0;
}

bool Sampler::getSample(Sample *s) {
    //printf("getSample i = %d, j = %d \n", i , j);
    if(i < width) {
        if (j < height-1) {
            Sample news = Sample();
            news.x = i + 0.5;
            news.y = j + 0.5;
            *s = news;
            i++;
            //printf("getSample news.x = %f, news.y = %f \n", news.x , news.y);
            return true;
        } else {
            return false;
        }
    } else {
        i = 0;
        j++;
        Sample news = Sample();
        news.x = i + 0.5;
        news.y = j + 0.5;
        *s = news;
        i++;
        //printf("getSample news.x = %f, news.y = %f \n", news.x , news.y);
        return true;
    }

}
//****************************************************
// More Global Variables
//****************************************************
string filename;
Camera eye;
FIBITMAP * bitmap;
vector<Shape *> scene_shapes;
vector<Light> scene_lights;


//****************************************************
// Image Writing
//****************************************************

void setPixel(int x, int y, Color rgb) {
    RGBQUAD color;
    color.rgbRed = rgb.r;
    color.rgbGreen = rgb.g;
    color.rgbBlue = rgb.b;
    FreeImage_SetPixelColor(bitmap, x, y, &color);
}


//****************************************************
// Render Loop
//****************************************************

void render() {
    Sample s = Sample();
    Sampler mySampler = Sampler();
    while(mySampler.getSample(&s)) {
        printf("sample generated at: %f, %f \n", s.x, s.y);
        //Ray r;
        //camera.generateRay(sample, &r;);
    }
}

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
    //printf("eaddd x: %.20f, y: %.20f, z: %.20f", eaddd.x, eaddd.y, eaddd.z);

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
  cout << "loading Scene .. \n"<< endl;
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
        printf("Outputting image of size: %d x %d\n", width, height);
      }
      //maxdepth depth
      //  max # of bounces for ray (default 5)
      else if(!splitline[0].compare("maxdepth")) {
        maxdepth = atoi(splitline[1].c_str());
        printf("Raytracing with maxdepth = %d\n", maxdepth);
      }
      //output filename
      //  output file to write image to 
      else if(!splitline[0].compare("output")) {
        filename = splitline[1];
        printf("Writing to file: %s\n", filename.c_str());
      }

      //camera lookfromx lookfromy lookfromz lookatx lookaty lookatz upx upy upz fov
      //  speciï¬es the camera in the standard way, as in homework 2.
      else if(!splitline[0].compare("camera")) {
        // lookfrom:
        float lfx  = atof(splitline[1].c_str());
        float lfy  = atof(splitline[2].c_str());
        float lfz  = atof(splitline[3].c_str());
        // lookat:
        float lax  = atof(splitline[4].c_str());
        float lay  = atof(splitline[5].c_str());
        float laz  = atof(splitline[6].c_str());
        // up:
        float upx  = atof(splitline[7].c_str());
        float upy  = atof(splitline[8].c_str());
        float upz  = atof(splitline[9].c_str());
        //fov:
        float fov = atof(splitline[10].c_str());

        eye.lookfrom = Point(lfx, lfy, lfz);
        eye.lookat = Point(lax, lay, laz);
        eye.up = Vector(upx, upy, upz);
        eye.fov = fov;
        printf("==== CAMERA ADDED ====\n");
        printf("lookfrom: \t %f, \t %f, \t %f \n", lfx, lfy, lfz);
        printf("lookat: \t %f, \t %f, \t %f \n", lax, lay, laz);
        printf("up vector: \t %f, \t %f, \t %f \n\n", upx, upy, upz);
      }

      //sphere x y z radius
      //  Deï¬nes a sphere with a given position and radius.
      else if(!splitline[0].compare("sphere")) {
        float x = atof(splitline[1].c_str());
        float y = atof(splitline[2].c_str());
        float z = atof(splitline[3].c_str());
        float r = atof(splitline[4].c_str());
        // Create new sphere:
        scene_shapes.push_back(&Sphere(Point(x, y, z), r));
        printf("==== SPHERE ADDED ====\n");
        printf("center: \t %f, \t %f, \t %f\n", x, y, z);
        printf("radius: \t %f \n", r);

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
  //string error = "no error";
  //bool vectest = testVector(&error);
  //printf("vectest returned with message: %s \n", error.c_str());
  //bool normaltest = testNormal(&error);
  //printf("normaltest returned with message: %s \n", error.c_str());

  loadScene(argv[1]);
  render();
  FreeImage_Initialise();
  cout << "FreeImage " << FreeImage_GetVersion() << "\n";
  cout << FreeImage_GetCopyrightMessage() << "\n\n";
  FIBITMAP * bitmap = FreeImage_Allocate(100, 100, 24);
  FreeImage_Save(FIF_PNG, bitmap, filename.c_str(), 0);
  printf("image sucessfully saved to %s\n", filename.c_str());
  FreeImage_DeInitialise();
}
