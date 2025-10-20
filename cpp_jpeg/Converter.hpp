#ifndef CONVERTER_HPP
#define CONVERTER_HPP

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp> // NOLINT
#include "Point3D.hpp"

using namespace cv;
using namespace std;

class Converter
{
private:
    // # Initalisation
    // self.image = np.array(Image.open(img_path))
    // self.taille_bloc = taille_bloc
    // self.quality = quality
    // self.mode = mode
    // self.H, self.W, self.nbr_canaux = (self.image).shape
    // self.prev_dc = 0
    // self.C = getC(self.taille_bloc)
    Mat image, C;
    int taille_bloc, H, W, nbr_canaux, quality, prev_dc;
    string mode;
    void init();

public:
    // Constructeur semi-complet
    Converter(Mat img, int taille_bloc);
    // Constructeur complet
    Converter(Mat img, int taille_bloc, int quality, int prev_dc, string mode);
};

#endif // CONVERTER_HPP