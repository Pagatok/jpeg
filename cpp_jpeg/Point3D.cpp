#include <cmath>
#include "Point3D.hpp"

Point3D::Point3D() : x(0), y(0), z(0) {}
Point3D::Point3D(float x, float y, float z) : x(x), y(y), z(z) {}

float Point3D::getX() const { return x; }
float Point3D::getY() const { return y; }
float Point3D::getZ() const { return z; }

void Point3D::setX(float x) { this->x = x; }
void Point3D::setY(float y) { this->y = y; }
void Point3D::setZ(float z) { this->z = z; }

void Point3D::afficher() const
{
    std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
}

float Point3D::norme() const
{
    return std::sqrt(x * x + y * y + z * z);
}

Point3D rgb2ycbcr(Point3D rgbpoint)
{

    float R = rgbpoint.getX();
    float G = rgbpoint.getY();
    float B = rgbpoint.getZ();

    float Y = 16 + (65.738 * R + 129.057 * G + 25.064 * B) / 256;
    float Cb = 128 - (37.945 * R + 74.494 * G + 112.439 * B) / 256;
    float Cr = 128 + (112.439 * R + 94.154 * G + 18.285 * B) / 256;

    Point3D p(Y, Cb, Cr);

    return (p);
}