#ifndef POINT3D_H
#define POINT3D_H

#include <iostream>

class Point3D
{
private:
    float x, y, z;

public:
    // Constructeurs
    Point3D();                          // constructeur par défaut
    Point3D(float x, float y, float z); // constructeur avec valeurs

    // Méthodes d'accès
    float getX() const;
    float getY() const;
    float getZ() const;

    void setX(float x);
    void setY(float y);
    void setZ(float z);

    // Méthodes diverses
    void afficher() const;
    float norme() const;
};

Point3D rgb2ycbcr(Point3D);

#endif