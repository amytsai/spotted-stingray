#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <stack>
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
#define EPSILON .0001f
#define airRefractIndex 1.0f
inline float sqr(float x) { return x*x; }

using namespace std;

//****************************************************
// Some Classes
//****************************************************
class Point;
class Vector;
class Normal;
class Ray;
class Transformation; //Created from MatrixGenerator, contains the matrix and inverse
class MatrixGenerator; //Generates the transformations
class Color;
class Camera;
class LocalGeo;
class Shape;
class Sphere;
class Triangle;
class Primitive;
class Intersection;
class GeometricPrimitive;
class Light;
class Sample;
class Sampler;
class BRDF; //Stores the coefficients for specular, ambient, diffuse, reflection
class Material; //I have no clue, but it's in the design document
class MTS; //Matrix Transformation Stack

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
	Point sub(Vector);
	Vector sub(Point); //Uses current point as the arrow side of vector
	Point transform(Transformation); //Returns the transformed point
};

//***************** VECTOR *****************//
class Vector {
public:
	Vector4f vector;
	float len;
	Vector();
	Vector(float, float, float);
	Vector(Point, Point); //Vector(start point, end point)
	Vector(Vector4f);
	Vector add(Vector);
	Vector sub(Vector);
	Vector mult(float);
	Vector div(float);
	float dot(Vector);
	Vector cross(Vector);
	void normalize();
	bool equals(Vector);
	Vector transform(Transformation); //Returns the transformed vector
};

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
	float dot(Normal);
	bool equals(Normal);
	Normal transform(Transformation); //Returns the transformed normal vector. Only use this with normals, not with vectors!
};

//***************** RAY *****************//
class Ray {
	/* Represents the ray:
	r(t) = pos + t*dir*/
public:
	Point pos;
	Vector dir;
	float t_min, t_max;
	Ray();
	Ray(Point, Point);
	Ray(Point, Vector);
	Ray(Point, Vector, float);
	Point getPoint(float); //Get's the value of the ray at the input time t (pos + t * dir)
	Ray transform(Transformation); //Returns the transformed ray
};

//***************** TRANSFORMATION *****************//
class Transformation {
public: 
	Matrix4f matrix, matrixinv, matrixinvtrans; //matrix, inverse matrix, and inverse transposed matrix
	Transformation(); //Do not use
	Transformation(Matrix4f);
	Transformation multOnRightSide(Transformation); //returns Transformation matrix of other * this
	Transformation multOnLeftSide(Transformation); //returns Transformation matrix of this * other
	//Remember, for combining transformations, the first transformation should be on the right side. Transformation RS applies S first, R second
	Transformation generateInverse();
};

//***************** MATRIX *****************//
class MatrixGenerator {
public:
	MatrixGenerator();
	Transformation generateTranslation(float, float, float); //Translation transform matrix, pass in the x, y, z translation triplet
	Transformation generateRotationx(float); //Rotation around x axis matrix, pass in degrees
	Transformation generateRotationy(float); //Rotation around y axis matrix, pass in degrees
	Transformation generateRotationz(float); //Rotation around z axis matrix, pass in degrees
	Transformation generateScale(float, float, float); //Scale transform matrix, pass in the x, y, z scaling triplet
	Transformation generateIdentity();
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
	Color mult(Color); //Used only for multiple shading constants with a color
	Color div(float);
	void clamp();
};

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
	void generateRay(Sample, Ray*);
};

//***************** LOCAL GEO *****************//
class LocalGeo {
public:
	Point pos;
	Normal n;
	LocalGeo();
	LocalGeo(Point, Normal);
	LocalGeo transform(Transformation); //Applies the transformation matrix on the Local Geo, transforming both the point and the normal
};

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
	bool intersect(Ray&, float*, LocalGeo*); //Returns whether it intersects, the time it hits, and the point/normal
	bool ifIntersect(Ray&); //Returns whether it intersects
};

//***************** TRIANGLE *****************//
class Triangle : public Shape {
public:
	Point r, s, t;
	Normal norm;
	bool vertexNormal;
	Triangle(Point, Point, Point);
	bool intersect(Ray&, float* , LocalGeo* ); //Returns whether it intersects, the time it hits, and the point/normal
	bool ifIntersect(Ray& ); //Returns whether it intersects
};

//***************** PRIMITIVE *****************//
class Primitive {
public:
	virtual bool intersect(Ray&, float*, Intersection* ) = 0;
	virtual bool ifIntersect(Ray& ) = 0;
	virtual void getBRDF(LocalGeo&, BRDF*) = 0;
};

//***************** INTERSECTION *****************//
class Intersection {
public:
	LocalGeo localGeo;
	Primitive* primitive;
	Intersection();
	Intersection(LocalGeo, Primitive*);
};

//***************** GEOMETRIC PRIMITIVE *****************//

//POTENTIAL ERROR: We made need to reverse objToWorld and worldToObj
class GeometricPrimitive : public Primitive {
public:
	Transformation objToWorld, worldToObj;
	Shape* shape;
	Material* mat;
	GeometricPrimitive(Shape*, Material*, Transformation); 
	//We combine all the transformations together first using the transformation multiplication functions before inputting
	GeometricPrimitive(Shape*, Material*);  //Defaults to identity transformation matrix
	bool intersect(Ray&, float*, Intersection*);
	bool ifIntersect(Ray&);
	void getBRDF(LocalGeo&, BRDF*);
};

//***************** LIGHT *****************//
class Light {
public:
	float x, y, z;
	Vector direction;
	Color rgb;
	bool isPL;
	Light();
	Light(float, float, float, Color, bool); //Point light constructor
	Light(float, float, float, Color, bool, Vector); //Directional light constructor
	void generateLightRay(LocalGeo&, Ray*, Color*); 
	void generateShadowRay(LocalGeo&, Ray*, Color*); 
	Light transform(Transformation);
};

//***************** SAMPLE *****************//
class Sample {
public:
	//holds screen coordinates;
	float x, y;
	Sample();
};

//***************** SAMPLER *****************//
class Sampler {
public:
	int i, j;
	bool getSample(Sample *);
	Sampler();
};

//****************************************************
// BRDF
//****************************************************

class BRDF {
public:
	Color kd, ks, ka, kr, ke; //All the constants for shading
	float refr, refrIndex; //Refraction constant
	BRDF();
	BRDF(Color, Color, Color, Color);
};

//****************************************************
// MATERIAL
//****************************************************

class Material {
public:
	BRDF constantBRDF;
	Material();
	Material(BRDF);
	void getBRDF(LocalGeo& local, BRDF* brdf);
};


//****************************************************
// MTS
//****************************************************
class MTS {
public:
	stack<Transformation> tStack;
	MTS();
	MTS(Transformation);
	void push(Transformation);
	void pop();
	Transformation top();
	Transformation evaluateStack();
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
	point = Vector4f(vec(0), vec(1), vec(2), 1);
}

Point Point::add(Vector v) {
	Vector4f temp = point + v.vector;
	return Point(temp);
}

Point Point::sub(Vector v) {
	Vector4f temp = point - v.vector;
	return Point(temp);
}

Vector Point::sub(Point p) {
	Vector4f temp = point - p.point;
	return Vector(temp);
}

Point Point::transform(Transformation trans) {
	Point temp;
	temp = Point(trans.matrix * point);
	return temp;
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
	vector = Vector4f(vec(0), vec(1), vec(2), 0);
	len = vector.norm();
}

Vector::Vector(Point start, Point end) {
	vector = end.point - start.point;
	vector = Vector4f(vector(0), vector(1), vector(2), 0);
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
	len = 1.0;
}

bool Vector::equals(Vector v) {
	Vector4f temp = v.vector - vector;
	float size = temp.norm();
	return size == 0;
}

Vector Vector::transform(Transformation trans){
	Vector temp;
	temp = Vector(trans.matrix * vector);
	return temp;
}


//***************** NORMAL METHODS*****************//

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
	normal = Vector4f(v.vector(0), v.vector(1), v.vector(2), 0);
	normal.normalize();
}
Normal::Normal(Vector4f vec) {
	normal = Vector4f(vec(0), vec(1), vec(2), 0);
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

float Normal::dot(Normal v) {
	return normal.dot(v.normal);
}

bool Normal::equals(Normal n) {
	Vector4f temp = n.normal - normal;
	float size = temp.norm();
	return size == 0;
}

Normal Normal::transform(Transformation trans) {
	Normal temp = Normal(trans.matrixinvtrans * normal);
	return temp;
}


//***************** RAY METHODS*****************//

Ray::Ray(Point a, Point b) {
	pos = Point(a.point);
	dir = Vector(a, b);
	t_min = 0;
	t_max = 99999999;
}

Ray::Ray(Point a, Vector v) {
	pos = Point(a.point);
	dir = Vector(v.vector);
	t_min = 0;
	t_max = 99999999;
}

Ray::Ray(Point a, Vector v, float t) {
	pos = Point(a.point);
	dir = Vector(v.vector);
	t_min = t;
	t_max = 99999999;
}
Ray::Ray() {
	pos = Point();
	dir = Vector();
	t_min = 0;
	t_max = 99999999;
}

Point Ray::getPoint(float time) {
	Vector travel = dir.mult(time);
	Point temp;
	temp = pos.add(travel);
	return temp;
}

Ray Ray::transform(Transformation trans) {
	Point newPoint = pos.transform(trans);
	Vector newDir = dir.transform(trans);
	return Ray(newPoint, newDir);
}

//***************** TRANSFORMATION METHODS*****************//

Transformation::Transformation(Matrix4f mat) {
	matrix = mat;
	matrixinv = mat.inverse();
	matrixinvtrans = matrixinv.transpose();
}

Transformation::Transformation() {

}

Transformation Transformation::multOnRightSide(Transformation other) {
	Transformation temp = Transformation(other.matrix * matrix);
	return temp;
}

Transformation Transformation::multOnLeftSide(Transformation other) {
	Transformation temp = Transformation(matrix * other.matrix);
	return temp;
}

Transformation Transformation::generateInverse() {
	Transformation temp = Transformation(matrixinv);
	return temp;
}


//***************** MATRIX METHODS*****************//

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

Transformation MatrixGenerator::generateIdentity() {
	Matrix4f temp;
	temp << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;
	return temp;
}


//***************** COLOR METHODS *****************//

Color::Color() {
	r = 0.0f;
	g = 0.0f;
	b = 0.0f;
}

Color::Color(float red, float green, float blue) {
	r = red;
	g = green;
	b = blue;
}

Color Color::add(Color v) {
	float a = r + v.r;
	float s = g + v.g;
	float c = b + v.b;

	return Color(a,s,c);
}

Color Color::sub(Color v) {
	float a = r - v.r;
	float s = g - v.g;
	float c = b - v.b;

	return Color(a,s,c);
}

Color Color::mult(float k) {
	float a = k*r;
	float s = k*g;
	float c = k*b;

	return Color(a,s,c);
}

Color Color::mult(Color col) {
	return Color(r * col.r, g * col.g, b * col.b);
}

Color Color::div(float k) {
	float a = r/k;
	float s = g/k;
	float c = b/k;

	return Color(a,s,c);
}

void Color::clamp() {
	r = min(r, 1.0f);
	g = min(g, 1.0f);
	b = min(b, 1.0f);
}


//***************** CAMERA METHODS *****************//

Camera::Camera() {
	lookfrom = Point();
	lookat = Point();
	up = Vector();
	fov = 0.0;
}

Camera::Camera(Point from, Point at, Vector v, float f) {
	lookfrom = from;
	lookat = at;
	Vector offset = Vector(from, Point(0, 0, 0));

	up = v;
	up.normalize();
	fov = f;
	//Vector y  = up.mult(tan(fov/2));

	Vector z = Vector(from, at);
	z.normalize();

	Vector x = up.cross(z);
	x.normalize();

	Vector y = z.cross(x);
	y.normalize();
	y = y.mult(tan(fov/2));
	float w = (float) width;
	float h = (float) height;
	x = x.mult((w/h) * y.len);
	/*printf("x = %f, y = %f, z = %f\n", x.len, y.len, z.len);
	printf("x <%f, %f, %f> \n", x.vector(0), x.vector(1), x.vector(2));
	printf("z <%f, %f, %f> \n", z.vector(0), z.vector(1), z.vector(2));
	printf("z <%f, %f, %f> \n", z.vector(0), z.vector(1), z.vector(2));*/
	Vector zminusx = z.sub(x);
	Vector zplusx = z.add(x);

	UL = zminusx.add(y);
	LL = zminusx.sub(y);
	UR = zplusx.add(y);
	LR = zplusx.sub(y);

	UL = UL.sub(offset);
	LL = LL.sub(offset);
	UR = UR.sub(offset);
	LR = LR.sub(offset);

	printf("UL <%f, %f, %f> \n", UL.vector(0), UL.vector(1), UL.vector(2));
	printf("LL <%f, %f, %f> \n", LL.vector(0), LL.vector(1), LL.vector(2));
	printf("UR <%f, %f, %f> \n", UR.vector(0), UR.vector(1), UR.vector(2));
	printf("LR <%f, %f, %f> \n", LR.vector(0), LR.vector(1), LR.vector(2));

}



//***************** LOCAL GEO METHODS *****************//

LocalGeo::LocalGeo() {
	pos = Point();
	n = Normal();
}
LocalGeo::LocalGeo(Point p, Normal norm) {
	pos = Point(p.point);
	n = Normal(norm.normal);
}

LocalGeo LocalGeo::transform(Transformation trans) {
	Point temppoint = pos.transform(trans);
	Normal tempnorm = n.transform(trans);
	return LocalGeo(temppoint, tempnorm);
}


//***************** SPHERE METHODS *****************//


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
		float hittime1 = (-d.dot(e - c) + sqrt(determinant))/(d.dot(d));
		float hittime2 = (-d.dot(e - c) - sqrt(determinant))/(d.dot(d));
		float hittime = min(hittime1, hittime2);
		*thit = hittime;
		if(hittime < ray.t_min || hittime > ray.t_max) {
			return false;
		}
		Point hitPoint = ray.getPoint(hittime);
		Normal norm = Normal((hitPoint.sub(center)));
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
	float determinant = pow(d.dot(e - c), 2) - (d.dot(d))*((e - c).dot(e - c) - r*r);
	if(pow(d.dot(e - c), 2) - (d.dot(d))*((e - c).dot(e - c) - r*r) < 0) {
		return false;
	}
	else {
		float hittime1 = (-d.dot(e - c) + sqrt(determinant))/(d.dot(d));
		float hittime2 = (-d.dot(e - c) - sqrt(determinant))/(d.dot(d));
		float hittime = min(hittime1, hittime2);
		if(hittime < ray.t_min || hittime > ray.t_max) {
			return false;
		}
		return true;
	}
}


//***************** TRIANGLE METHODS *****************//

Triangle::Triangle(Point first, Point second, Point third) {
	r = first;
	s = second;
	t = third;
	vertexNormal = false;
	Vector rs= s.sub(r);
	Vector rt = t.sub(r);
	norm = Normal(rs.cross(rt));
}

bool Triangle::intersect(Ray& ray, float* thit, LocalGeo* local) {
	//cout << "r =" << endl << r.point << endl;
	//cout << "s =" << endl << s.point << endl;
	//cout << "t =" << endl << t.point << endl;
	/*printf("raystart \n");
	cout << ray.pos.point << endl;
	printf("ray direction \n");
	cout << ray.dir.vector << endl;*/
	Point rayStart = ray.pos;
	Vector rayDirection = ray.dir;
	Vector4f av = r.point;
	Vector4f bv = s.point;
	Vector4f cv = t.point;
	Vector4f dv = rayDirection.vector;
	Vector4f ev = rayStart.point;
	float a, b, c, d, e, f, g, h, i, j, k, l, M;
	float beta, gamma, hittime;
	a = av(0) - bv(0);
	b = av(1) - bv(1);
	c = av(2) - bv(2);
	d = av(0) - cv(0);
	e = av(1) - cv(1);
	f = av(2) - cv(2);
	g = dv(0);
	h = dv(1);
	i = dv(2);
	j = av(0) - ev(0);
	k = av(1) - ev(1);
	l = av(2) - ev(2);
	M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);
	//printf("Value of a, b, c: %f, %f, %f \n", a, b, c);
	//printf("Value of d, e, f: %f, %f, %f \n", d, e, f);
	//printf("Value of g, h, i: %f, %f, %f \n", g, h, i);
	//printf("Value of j, k, l: %f, %f, %f \n", j, k, l);
	//printf("Value of M: %f \n", M);
	hittime = (-1)*(f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/M;
	//if(hittime < ray.t_min || hittime > ray.t_max) {
	if(hittime < ray.t_min || hittime > ray.t_max) {
		//printf("Fail on hittime check: %f \n", hittime);
		return false;
	}
	gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
	if(gamma < 0 || gamma > 1) {
		//printf("Fail on gamma check: %f \n", gamma);
		return false;
	}
	beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
	if(beta < 0 || beta > (1- gamma)) {
		//printf("Fail on beta check: %f \n", beta);
		return false;
	}

	else {
		*thit = hittime;
		Vector temp = Vector(norm.normal);
		if(temp.dot(rayDirection) > 0) {
			temp = temp.mult(-1);
		}
		Vector rs= s.sub(r);
		Vector rt = t.sub(r);
		Point intersectionPoint = r.add(rs.mult(beta));
		intersectionPoint = intersectionPoint.add(rt.mult(gamma));
		//*local = LocalGeo(intersectionPoint, Normal(temp));
		*local = LocalGeo(ray.getPoint(hittime), Normal(temp));
		printf("The point of intersection: (%f, %f, %f) \n", (*local).pos.point(0), (*local).pos.point(1), (*local).pos.point(2));
		printf("The normal of intersection: <%f, %f, %f> \n", (*local).n.normal(0), (*local).n.normal(1), (*local).n.normal(2));
		return true;
	}
}

bool Triangle::ifIntersect(Ray& ray) {
	Point rayStart = ray.pos;
	Vector rayDirection = ray.dir;
	Vector4f av = r.point;
	Vector4f bv = s.point;
	Vector4f cv = t.point;
	Vector4f dv = rayDirection.vector;
	Vector4f ev = rayStart.point;
	float a, b, c, d, e, f, g, h, i, j, k, l, M;
	float beta, gamma, hittime;
	a = av(0) - bv(0);
	b = av(1) - bv(1);
	c = av(2) - bv(2);
	d = av(0) - cv(0);
	e = av(1) - cv(1);
	f = av(2) - cv(2);
	g = dv(0);
	h = dv(1);
	i = dv(2);
	j = av(0) - ev(0);
	k = av(1) - ev(1);
	l = av(2) - ev(2);
	M = a*(e*i - h*f) + b*(g*f - d*i) + c*(d*h - e*g);
	hittime = (-1)*(f*(a*k - j*b) + e*(j*c - a*l) + d*(b*l - k*c))/M;
	//if(hittime < ray.t_min || hittime > ray.t_max) {
	if(hittime < ray.t_min || hittime > ray.t_max) {
		//printf("Fail on hittime check: %f \n", hittime);
		return false;
	}
	gamma = (i*(a*k - j*b) + h*(j*c - a*l) + g*(b*l - k*c))/M;
	if(gamma < 0 || gamma > 1) {
		//printf("Fail on gamma check: %f \n", gamma);
		return false;
	}
	beta = (j*(e*i - h*f) + k*(g*f - d*i) + l*(d*h - e*g))/M;
	if(beta < 0 || beta > (1- gamma)) {
		//printf("Fail on beta check: %f \n", beta);
		return false;
	}
	else {
		return true;
	}
}

//***************** LIGHT METHODS *****************//

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

Light::Light(float a, float b, float c, Color color, bool PL, Vector dir) {
	x = a;
	y = b;
	z = c;
	rgb = color;
	isPL = PL;
	direction = dir;
}

void Light::generateLightRay(LocalGeo& local, Ray* lray, Color* lcolor) {
	if(isPL) {
		Point origin = Point(x, y, z);
		Vector dir = Vector(local.pos, origin);
		*lray = Ray(origin, dir);
		*lcolor = rgb;
		return;
	}
	else {		
		Vector dir = Vector(x, y, z);
		dir = dir.mult(-1);
		Point origin = local.pos;
		*lray = Ray(origin, dir);
		*lcolor = rgb;
		return;
	}
}

void Light::generateShadowRay(LocalGeo& local, Ray* lray, Color* lcolor) {
	if(isPL) {
		Point origin = Point(x, y, z);
		Vector dir = Vector(local.pos, origin);
		*lray = Ray(origin, dir, EPSILON);
		*lcolor = rgb;
		return;
	}
	else {		
		Vector dir = Vector(x, y, z);
		dir = dir.mult(-1);
		Point origin = local.pos;
		*lray = Ray(origin, dir, EPSILON);
		*lcolor = rgb;
		return;
	}
}

Light Light::transform(Transformation trans) {
	Point tempPoint = Point(x, y, z);
	tempPoint = tempPoint.transform(trans);
	return Light(tempPoint.point(0), tempPoint.point(1), tempPoint.point(2), rgb, isPL, direction.transform(trans));
}

//***************** SAMPLE METHODS *****************//

Sample::Sample() {
	x, y = 0.0;
}

void Camera::generateRay(Sample s, Ray* ray) {
	float imagePlaneW = (UL.sub(UR)).len;
	float imagePlaneH = (UL.sub(LL)).len;
	float imgToScreen = imagePlaneW/width;
	float u = (s.x - (((float) width) / 2))*imgToScreen + imagePlaneW/2;
	float v = -((s.y - (((float) height) / 2))*imgToScreen) + imagePlaneH/2;
	//printf("value of v and u: %f, %f \n", v, u);
	v = v/imagePlaneH;
	u = u/imagePlaneW;
	Vector t1 = LL.mult(v).add(UL.mult(1-v));
	Vector t2 = LR.mult(v).add(UR.mult(1-v));
	Vector t3 = t1.mult(u).add(t2.mult(1-u));
	//printf("value of the 3 vectors: v1<%f, %f, %f>, v2<%f, %f, %f>, v3<%f, %f, %f> \n", t1.vector(0), t1.vector(1), t1.vector(2), t2.vector(0), t2.vector(1), t2.vector(2), t3.vector(0), t3.vector(1), t3.vector(2));
	Point P = Point(t3.vector);
	*ray  = Ray(lookfrom, P);
}


//***************** SAMPLER METHODS *****************//

Sampler::Sampler() {
	i, j = 0;
}

bool Sampler::getSample(Sample *s) {
	//printf("getSample i = %d, j = %d \n", i , j);
	if(i < width) {
		if (j < height) {
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
	} else if(j <  height-1){
		i = 0;
		j++;
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

}
//****************************************************
// More Global Variables
//****************************************************
string filename;
Camera eye = Camera();
FIBITMAP * bitmap;
typedef vector<Primitive*> shape_list;
shape_list* l = new shape_list();
typedef vector<Light*> light_list;
light_list* lightsList = new light_list();

//****************************************************
// Image Writing
//****************************************************

void setPixel(int x, int y, Color rgb) {
	RGBQUAD color;
	color.rgbRed = max(min(rgb.r*255, 255.0f), 0.0f);
	color.rgbGreen = max(min(rgb.g*255, 255.0f), 0.0f);
	color.rgbBlue = max(min(rgb.b*255, 255.0f), 0.0f);
	FreeImage_SetPixelColor(bitmap, x, y, &color);
}

//***************** BRDF METHODS *****************//


BRDF::BRDF() {
	kd = Color();
	ks = Color();
	ka = Color();
	kr = Color();
}

BRDF::BRDF(Color d, Color s, Color a, Color r) {
	kd = d;
	ks = s;
	ka = a;
	kr = r;
}



//***************** MATERIAL METHODS *****************//

Material::Material() {
	constantBRDF = BRDF();
}

Material::Material(BRDF mat) {
	constantBRDF = mat;
}

void Material::getBRDF(LocalGeo& local, BRDF* brdf) {
	*brdf = constantBRDF;
}

//***************** INTERSECTION METHODS *****************//


Intersection::Intersection() {

}

Intersection::Intersection(LocalGeo loc, Primitive* prim) {
	localGeo = loc;
	primitive = prim;
}

//***************** GEOMETRICPRIMITIVE METHODS *****************//

GeometricPrimitive::GeometricPrimitive(Shape* objshape, Material* objmat, Transformation trans) {
	shape = objshape;
	mat = objmat;
	//worldToObj = trans;
	//objToWorld = worldToObj.generateInverse();
	objToWorld = trans;
	worldToObj = objToWorld.generateInverse();
}

GeometricPrimitive::GeometricPrimitive(Shape* objshape, Material* objmat) {
	MatrixGenerator temp = MatrixGenerator();
	shape = objshape;
	mat = objmat;
	//worldToObj = temp.generateIdentity();
	//objToWorld = worldToObj.generateInverse();
	objToWorld = temp.generateIdentity();
	worldToObj = objToWorld.generateInverse();
}

bool GeometricPrimitive::intersect(Ray& ray, float* thit, Intersection* in) {
	/*printf("world to Obj\n");
	cout << worldToObj.matrix << endl;
	printf("Obj to world\n");
	cout << objToWorld.matrix << endl;*/
	Ray objectRay = ray.transform(worldToObj);
	LocalGeo objectLocalGeo;   
	if(shape->intersect(objectRay, thit, &objectLocalGeo)) {
		in->primitive = this;
		in->localGeo = objectLocalGeo.transform(objToWorld);
		return true;    
	}
	else {
		return false;
	}         
}

bool GeometricPrimitive::ifIntersect(Ray& ray) {
	Ray objectRay = ray.transform(worldToObj);
	return (*shape).ifIntersect(objectRay);
}

void GeometricPrimitive::getBRDF(LocalGeo& local, BRDF* brdf) {
	(*mat).getBRDF(local, brdf);
}


//***************** MTS METHODS  *****************//
MTS::MTS() {
	MatrixGenerator m = MatrixGenerator();
	tStack.push(m.generateIdentity());
}

MTS::MTS(Transformation t) {
	tStack.push(t);
}

void MTS::push(Transformation t) {
	tStack.push(t);
}

void MTS::pop() {
	tStack.pop();
}

Transformation MTS::top() {
	return tStack.top();
}

Transformation MTS::evaluateStack() {
	MatrixGenerator m = MatrixGenerator();
	Transformation curTransformation = m.generateIdentity();
	while(!tStack.empty()) {
		curTransformation = curTransformation.multOnRightSide(tStack.top());
		tStack.pop();
	}
	tStack.push(curTransformation);
	return curTransformation;
}

//****************************************************
// Ray Tracer TRACE
//****************************************************

//THE VECTOR OF LIGHTS NEEDS TO BE CALLED lightsList


//For a ray, returns the time, Intersection object, and whether it actually hits anything for the scene
void findIntersection(Ray& ray, float* minTime, Intersection* minIntersect, bool* isHit) {
	float thit = 0.0f;
	Intersection curIntersect = Intersection();
	Primitive* primitivePtr;
	for(int x = 0; x < l->size(); x++) {
		//This loop finds the object hit first, then returns the hittime and intersection object
		//GET THE PRIMITIVE FROM THE VECTOR LINE GOES HERE
		primitivePtr = (*l)[x];
		bool intersects = (*primitivePtr).intersect(ray, &thit, &curIntersect);
		if(intersects) {
			*isHit = true;
			if(thit < *minTime) {
				*minTime = thit;
				(*minIntersect) = Intersection(curIntersect.localGeo, curIntersect.primitive);
			}
		}
	}
}

bool isShadowIntersection(Ray& ray, float* minTime, Intersection* minIntersect, bool* isHit) {
	float thit = 0.0f;
	Intersection curIntersect = Intersection();
	Primitive* primitivePtr;
	for(int x = 0; x < l->size(); x++) {
		//This loop finds the object hit first, then returns the hittime and intersection object
		//GET THE PRIMITIVE FROM THE VECTOR LINE GOES HERE
		primitivePtr = (*l)[x];
		bool intersects = (*primitivePtr).intersect(ray, &thit, &curIntersect);
		if(intersects) {
			*isHit = true;
			if(thit < *minTime) {
				*minTime = thit;
				(*minIntersect) = Intersection(curIntersect.localGeo, curIntersect.primitive);
			}
		}
	}
	return *isHit;
}



//Generates the reflected ray from a surface given the incoming ray and the surface. Probably going to need the add the epsilon fuzz factor
Ray createReflectRay(LocalGeo& localGeo, Ray& ray) {
	Vector4f d = ray.dir.vector;
	Vector4f n = localGeo.n.normal;
	Vector4f temp = d - 2*(d.dot(n))*n;
	//Move the start point a bit ahead to avoid float errors
	Ray reflectRay = Ray(localGeo.pos, Vector(temp), EPSILON);
	return reflectRay;
}

//THE SHADING FUNCTIONS FOR AMBIENT SPECULAR AND DIFFUSE GO HERE, WE MIGHT HAVE TO SPLIT THE FUNCTION FOR DIFFUSE AND SPECULAR
//localGeo = the point and the normal of the relevant location
//brdf = all the relevant variables (kd, ks, ka, kr)
//lray = the ray of light
//lcolor = color of the light
Color shading(LocalGeo& localGeo, BRDF& brdf, Ray& lray, Ray& ray, Color& lcolor, Color* cumSpecular) {
	float kr = brdf.kr.r;
	Color returnColor = Color(); //Begins at (0,0,0)
	Color I = lcolor;
	Color kd, ks;
	kd = brdf.kd;
	ks = brdf.ks;
	Normal n = localGeo.n;
	Normal l = Normal(lray.dir);
	Normal v = Normal(ray.dir.mult(-1));
	/*cout << "point = " << endl << localGeo.pos.point << endl;
	cout << "n =" << endl << n.normal << endl;
	cout << "l =" << endl << l.normal << endl;
	cout << "v =" << endl << v.normal << endl;*/

	//Diffuse shading
	Color diffuse = kd.mult(I.mult(max(0.0f, n.dot(l))));
	returnColor = returnColor.add(diffuse);

	//Specular shading YES PHONG CONSTANT
	Vector temp1 = Vector(l.normal * - 1);
	Vector temp2 = Vector(n.normal * (2 * l.dot(n)) );
	Normal r = Normal(temp1.add(temp2));
	Color specular = ks.mult(I.mult(pow(max(0.0f, v.dot(r)), kr)));
	Normal h = v.add(l);
	//Color specular = ks.mult(I.mult(pow(max(0.0f, n.dot(h)), kr)));

	returnColor = returnColor.add(specular);
	returnColor.clamp();
	return returnColor;
}

void trace(Ray& ray, int depth, Color* color) {
	bool isHit = false;
	float minTime = 99999999;
	Intersection minIntersect = Intersection();
	float thit = 0.0f;
	BRDF brdf = BRDF();
	printf("tracing ray with pos (%f, %f, %f) and dir <%f, %f, %f>\n", ray.pos.point(0), ray.pos.point(1), ray.pos.point(2), ray.dir.vector(0), ray.dir.vector(1), ray.dir.vector(2));
	if (depth > maxdepth) {
		Color temp = Color(0, 0, 0);
		*color = temp;
		return;
	}
	else {
		//BEGIN NEW CODE
		findIntersection(ray, &minTime, &minIntersect, &isHit);
		if(!isHit) { //Checks if we actually hit any objects, if we didn't then we return black
			Color temp = Color(0, 0, 0);
			*color = temp;
			return;
		}
		minIntersect.primitive->getBRDF(minIntersect.localGeo, &brdf);

		//SHADING BEGINS HERE
		Ray lray = Ray();
		Ray shadowRay = Ray();
		Color lcolor = Color();
		Color shadowColor = Color();
		Color cumSpecular = Color();
		bool lisHit = false;
		float lminTime = 99999999;
		Intersection lminIntersect = Intersection();

		//We do ambient and emissive shading here
		Color ka = brdf.ka;
		Color ke = brdf.ke;
		(*color) = (*color).add(ka);
		(*color) = (*color).add(ke);

		//We do diffuse, specular, and shadows here
		for (int i = 0; i < lightsList->size(); i++) {
			(*lightsList)[i]->generateLightRay(minIntersect.localGeo, &lray, &lcolor);
			(*lightsList)[i]->generateShadowRay(minIntersect.localGeo, &shadowRay, &shadowColor);
			bool isShadow = isShadowIntersection(shadowRay, &lminTime, &lminIntersect, &lisHit);
			if(!isShadow) {
				*color = (*color).add(shading(minIntersect.localGeo, brdf, lray, ray, lcolor, &cumSpecular));
			}						
		}

		// Handle mirror reflection
		//Checks to make sure that the reflection constant isn't (0, 0, 0)
		if (brdf.ks.r > 0 || brdf.ks.g > 0 || brdf.ks.b > 0) {
			Ray reflectRay = createReflectRay(minIntersect.localGeo, ray);
			// Make a recursive call to trace the reflected ray
			Color tempColor = Color();
			trace(reflectRay, depth+1, &tempColor);
			//*color = (*color).add(tempColor.mult(cumSpecular));
			*color = (*color).add(tempColor.mult(brdf.ks)); // Amy's fix
		}

		/*
		// Handles refraction, index of refraction of air is called airRefractIndex, we might need to pass in index of refraction each time
		// Also might need to rewrite sphere to support normals that point inward
		float refr = brdf.refr;
		if (refr > 0) {
			float rindex = brdf.refrIndex;
			float n = currentIndex/rindex;
			Normal N = Normal(minIntersect.localGeo.n.normal);
			Normal normRay = Normal(ray.dir);
			float cosI = (-1)* N.dot(normRay);
			float cosT2 = 1.0f - n * n * (1.0f - cosI * cosI);
			if (cosT2 > 0.0f)
			{
				Vector T = Vector(ray.dir);
				T.normalize();
				T = T.mult(n);
				Vector temp = Vector(N.normal);
				temp = temp.mult((n * cosI - sqrtf( cosT2 )));
				T = T.add(temp);
				Color tempColor = Color();
				Ray refractRay = Ray(minIntersect.localGeo.pos, T, EPSILON);
				trace(refractRay, depth+1, &tempColor);
				//Need the color of the material, not sure what it is, distance = distance traveled through object
				float dist = ray.dir.mult(minTime).len;
				Color absorbance = brdf.ke.mult(0.15f).mult(-dist);
				Color transparency = Color( expf( absorbance.r ), 
					expf( absorbance.g ), 
					expf( absorbance.b ) );
				//multiply by transparency like everything else
				*color = (*color).add(tempColor);
			}
		}
		*/
		//END NEW CODE
		/*
		//OLD CODE
		for (int i = 0; i < l->size(); i++ ) {
		Primitive* primitive = (*l)[i];
		Intersection inter;
		bool intersects = primitive->intersect(ray, &thit, &inter);
		if(intersects) {
		printf("===== HIT =====\n");
		//LocalGeo localgeo = intersect.localGeo;
		//Color temp = Color((localGeo.pos.point(0) + 1)/2, (localGeo.pos.point(1) + 1)/2, (localGeo.pos.point(2) + 1)/2);
		//Color temp = Color(0, 0,(inter.localGeo.pos.point(2) + 1)/2);
		Color temp = Color((inter.localGeo.pos.point(0) + 1)/5, 0,0);
		//Color temp = Color(1, 0, 0);
		//*color = temp;


		//Test shading code
		inter.primitive->getBRDF(inter.localGeo, &brdf);
		Ray lray = Ray();
		Ray shadowRay = Ray();
		Color lcolor = Color();
		Color shadowColor = Color();
		bool lisHit = false;
		float lminTime = 99999999;
		Intersection lminIntersect = Intersection();
		//We do ambient and emissive shading here
		Color ka = brdf.ka;
		Color ke = brdf.ke;
		(*color) = (*color).add(ka);
		(*color) = (*color).add(ke);
		for (int i = 0; i < lightsList->size(); i++) {
		(*lightsList)[i]->generateLightRay(inter.localGeo, &lray, &lcolor);
		(*lightsList)[i]->generateShadowRay(inter.localGeo, &shadowRay, &shadowColor);
		bool isShadow = isShadowIntersection(shadowRay, &lminTime, &lminIntersect, &lisHit);
		if(true) {
		printf("WE'RE ACTUALLY SHADING\n");
		(*color) = (*color).add(shading(inter.localGeo, brdf, lray, ray, lcolor));
		}						
		}
		//End test shading code



		return;
		} 
		}
		printf("===== MISS ====\n");
		Color temp = Color(0, 0, 0);
		*color = temp;
		return;
		*/
		//END OLD CODE
	}
}


//****************************************************
// Render Loop
//****************************************************

void render() {
	Sample s = Sample();
	Sampler mySampler = Sampler();
	while(mySampler.getSample(&s)) {
		printf("sample generated at: %f, %f \n", s.x, s.y);
		Ray r;
		eye.generateRay(s, &r);
		//printf("ray generated with pos (%f, %f, %f) and dir <%f, %f, %f>\n", r.pos.point(0), r.pos.point(1), r.pos.point(2), r.dir.vector(0), r.dir.vector(1), r.dir.vector(2));
		Color c = Color();
		trace(r, 0, &c);
		printf("color returned: %f, %f, %f\n", c.r, c.g, c.b);
		setPixel(s.x, s.y, c);
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
	cout << "loading Scene .. "<< endl;
	ifstream inpfile(file.c_str());
	if(!inpfile.is_open()) {
		std::cout << "Unable to open file" << std::endl;
	} else {
		std::string line;
		MTS tStack;
		MTS tBuffer;
		MatrixGenerator m = MatrixGenerator();
        tBuffer.push(m.generateIdentity());
		vector<Point> points;
		BRDF * curBRDF = new BRDF();

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

			//size width height
			else if(!splitline[0].compare("size")) {
				width = atoi(splitline[1].c_str());
				height = atoi(splitline[2].c_str());
				bitmap = FreeImage_Allocate(width, height, 24);
				printf("Outputting image of size: %d x %d\n", width, height);
			}
			//maxdepth depth
			else if(!splitline[0].compare("maxdepth")) {
				maxdepth = atoi(splitline[1].c_str());
				printf("Raytracing with maxdepth = %d\n", maxdepth);
			}
			//output filename
			else if(!splitline[0].compare("output")) {
				filename = splitline[1];
				printf("Writing to file: %s\n", filename.c_str());
			}

			//camera lookfromx lookfromy lookfromz lookatx lookaty lookatz upx upy upz fov
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
				float fov = (PI/180)*atof(splitline[10].c_str()); // convert to radians

				Point lookfrom = Point(lfx, lfy, lfz);
				Point lookat = Point(lax, lay, laz);
				Vector up = Vector(upx, upy, upz);
                if(up.len == 0) {
                    printf("invalid up vector for camera\n");
                    exit(EXIT_FAILURE);
                }
				eye = Camera(lookfrom, lookat, up, fov);
				tStack = MTS(); // push identity matrix
				printf("==== Camera Added ====\n");
				printf("lookfrom: \t %f, \t %f, \t %f \n", lfx, lfy, lfz);
				printf("lookat: \t %f, \t %f, \t %f \n", lax, lay, laz);
				printf("up vector: \t %f, \t %f, \t %f \n\n", upx, upy, upz);
			}

			//sphere x y z radius
			else if(!splitline[0].compare("sphere")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				float r = atof(splitline[4].c_str());
				// Create new sphere:
				Transformation *curTransform;
				Transformation buffer = tBuffer.evaluateStack();
				curTransform = new Transformation (buffer.multOnRightSide(tStack.top()));
				Shape* sphere;
				sphere = new Sphere(Point(x, y, z), r);
				l->push_back(new GeometricPrimitive(sphere, new Material(*curBRDF), *curTransform));
				printf("==== Sphere Added ====\n");
				printf("center: \t %f, \t %f, \t %f\n", x, y, z);
				printf("radius: \t %f \n", r);
				//TODO:   Store current property values
			}

			//maxverts number
			else if(!splitline[0].compare("maxverts")) {
				// Care if you want
				// Or you can just use a STL vector, in which case you can ignore this
			}

			//maxvertnorms number
			else if(!splitline[0].compare("maxvertnorms")) {
				// Care if you want
			}

			//vertex x y z
			//  The vertex is put into a pile, starting to be numbered at 0.
			else if(!splitline[0].compare("vertex")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				Point vert = Point(x, y, z);
				points.push_back(vert);
			}

			//vertexnormal x y z nx ny nz
			//  Similar to the above, but define a surface normal with each vertex.
			//  The vertex and vertexnormal set of vertices are completely independent
			//  (as are maxverts and maxvertnorms).
			else if(!splitline[0].compare("vertexnormal")) {
				// x: atof(splitline[1].c_str()),
				// y: atof(splitline[2].c_str()),
				// z: atof(splitline[3].c_str()));
				// nx: atof(splitline[4].c_str()),
				// ny: atof(splitline[5].c_str()),
				// nz: atof(splitline[6].c_str()));
				// Create a new vertex+normal with these 6 values, store in some array
			}

			//tri v1 v2 v3
			else if(!splitline[0].compare("tri")) {
				int v1 = atoi(splitline[1].c_str());
				int v2 = atoi(splitline[2].c_str());
				int v3 = atoi(splitline[3].c_str());
				Shape* triangle;
				triangle = new Triangle(points[v1], points[v2], points[v3]);
				Transformation* trans;
				Transformation buffer = tBuffer.evaluateStack();
				trans = new Transformation(buffer.multOnRightSide(tStack.top()));
				printf("before add triangle\n");
				l->push_back(new GeometricPrimitive(triangle, new Material(*curBRDF), *trans));
				printf("==== triangle added ====\n");
				//printf("normal: \t %f, \t %f, \t %f\n", *triangle.norm.normal(0), *triangle.norm.normal(1), *triangle.norm.normal(2));
				// create new triangle:
				//todo:   store current property values
				//todo:   store current top of matrix stack
			}

			//trinormal v1 v2 v3
			//  same as above but for vertices specified with normals.
			//  in this case, each vertex has an associated normal, 
			//  and when doing shading, you should interpolate the normals 
			//  for intermediate points on the triangle.
			else if(!splitline[0].compare("trinormal")) {
				// v1: atof(splitline[1].c_str())
				// v2: atof(splitline[2].c_str())
				// v3: atof(splitline[3].c_str())
				// create new triangle:
				//   store pointer to array of vertices (different array than above)
				//   store 3 integers to index into array
				//   store current property values
				//   store current top of matrix stack
			}

			//translate x y z
			//  A translation 3-vector
			else if(!splitline[0].compare("translate")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				Transformation trans = m.generateTranslation(x, y, z);
				tBuffer.push(trans);
				printf("====== ADDED TRANSLATE ======\n");
				printf("TOP OF TRANSFORMATION STACK: \n");
				//cout << transformationStack->back()->top().matrix << endl;
			}

			//rotate x y z angle
			//  Rotate by angle (in degrees) about the given axis as in OpenGL.
			else if(!splitline[0].compare("rotate")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				float angle = atof(splitline[4].c_str());
				Transformation trans = m.generateIdentity();
				if(x == 1.0f) {
					trans = m.generateRotationx(angle);
					printf("====== ADDED ROTATION ======\n");
					printf("%f degrees about the x axis\n", angle);
				} else if (y == 1.0f) {
					trans = m.generateRotationy(angle);
					printf("====== ADDED ROTATION ======\n");
					printf("%f degrees about the y axis]\n", angle);
				} else if (z == 1.0f) {
					trans = m.generateRotationz(angle);
					printf("====== ADDED ROTATION ======\n");
					printf("%f degrees about the z axis\n", angle);
				}
				tBuffer.push(trans);
				printf("TOP OF TRANSFORMATION STACK: \n");
				//cout << transformationStack->back()->top().matrix << endl;
			}

			//scale x y z
			//  Scale by the corresponding amount in each axis (a non-uniform scaling).
			else if(!splitline[0].compare("scale")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
                if(x == 0.0 || y == 0.0 || z == 0.0) {
                    printf("invalid scaling argument\n");
                    exit(EXIT_FAILURE);
                }
				printf("====== ADDED SCALE ======\n");
				Transformation trans = m.generateScale(x, y, z);
				tBuffer.push(trans);
				printf("TOP OF TRANSFORMATION STACK: \n");
				//cout << transformationStack.top().top().matrix << endl;
				printf("reached here\n");
			}

			//pushTransform
			//  Push the current modeling transform on the stack as in OpenGL. 
			else if(!splitline[0].compare("pushTransform")) {
				printf("beginning of pushTransform()\n");
				Transformation buffer = tBuffer.evaluateStack();
				tBuffer = MTS();
				tStack.push(buffer.multOnRightSide(tStack.top()));
				printf("====== PUSH TRANSFORM ======\n");
				printf("TOP OF TRANSFORMATION STACK: \n");
				//cout << transformationStack.back()->top().matrix << endl;
				printf("reached end of PushTransform\n");
			}

			//popTransform
			//  Pop the current transform from the stack as in OpenGL. 
			//  The sequence of popTransform and pushTransform can be used if 
			//  desired before every primitive to reset the transformation 
			//  (assuming the initial camera transformation is on the stack as 
			//  discussed above).
			else if(!splitline[0].compare("popTransform")) {
				tBuffer = MTS();
				tStack.pop();
				printf("====== POP TRANSFORM ======\n");
				printf("TOP OF TRANSFORMATION STACK: \n");
				//cout << transformationStack->back()->top().matrix << endl;
			}

			//directional x y z r g b
			//  The direction to the light source, and the color, as in OpenGL.
			else if(!splitline[0].compare("directional")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				float r = atof(splitline[4].c_str());
				float g = atof(splitline[5].c_str());
				float b = atof(splitline[6].c_str());
				lightsList->push_back(new Light(x, y, z, Color(r, g, b), false));
			}
			//point x y z r g b
			//  The location of a point source and the color, as in OpenGL.
			else if(!splitline[0].compare("point")) {
				float x = atof(splitline[1].c_str());
				float y = atof(splitline[2].c_str());
				float z = atof(splitline[3].c_str());
				float r = atof(splitline[4].c_str());
				float g = atof(splitline[5].c_str());
				float b = atof(splitline[6].c_str());
				lightsList->push_back(new Light(x, y, z, Color(r, g, b), true));
			} 

			//attenuation const linear quadratic
			//  Sets the constant, linear and quadratic attenuations 
			//  (default 1,0,0) as in OpenGL.
			else if(!splitline[0].compare("attenuation")) {
				// const: atof(splitline[1].c_str())
				// linear: atof(splitline[2].c_str())
				// quadratic: atof(splitline[3].c_str())
			}

			//ambient r g b
			//  The global ambient color to be added for each object 
			//  (default is .2,.2,.2)
			else if(!splitline[0].compare("ambient")) {
				float r = atof(splitline[1].c_str());
				float g = atof(splitline[2].c_str());
				float b = atof(splitline[3].c_str());
				curBRDF->ka = Color(r, g, b);
			}

			//diï¬€use r g b
			//  speciï¬es the diï¬€use color of the surface.
			else if(!splitline[0].compare("diffuse")) {
				float r = atof(splitline[1].c_str());
				float g = atof(splitline[2].c_str());
				float b = atof(splitline[3].c_str());
				curBRDF->kd = Color(r, g, b);
			}
			//specular r g b 
			//  speciï¬es the specular color of the surface.
			else if(!splitline[0].compare("specular")) {
				float r = atof(splitline[1].c_str());
				float g = atof(splitline[2].c_str());
				float b = atof(splitline[3].c_str());
				curBRDF->ks = Color(r, g, b);
			}
			//shininess s
			//  speciï¬es the shininess of the surface.
			else if(!splitline[0].compare("shininess")) {
				float shininess = atof(splitline[1].c_str());
				curBRDF->kr = Color(shininess, shininess, shininess);
			}
			//emission r g b
			//  gives the emissive color of the surface.
			else if(!splitline[0].compare("emission")) {
				float r = atof(splitline[1].c_str());
				float g = atof(splitline[2].c_str());
				float b = atof(splitline[3].c_str());
				curBRDF->ke = Color(r, g, b);
				// Update current properties
			} else {
				std::cerr << "Unknown command: " << splitline[0] << std::endl;
			}
		}
	}
}
//****************************************************
// the usual stuff, nothing exciting here
//****************************************************
int main(int argc, char *argv[]) {
	//string error = "no error";
	//bool vectest = testvector(&error);
	//printf("vectest returned with message: %s \n", error.c_str());
	//bool normaltest = testnormal(&error);
	//printf("normaltest returned with message: %s \n", error.c_str());

	loadScene(argv[1]);
	FreeImage_Initialise();
	render();
	FreeImage_Save(FIF_PNG, bitmap, filename.c_str(), 0);
	printf("image sucessfully saved to %s\n", filename.c_str());
	FreeImage_DeInitialise();
}
