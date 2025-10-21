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
    Mat original_image, C;
    vector<vector<Point3D>> work_image;
    int taille_bloc, H, W, nbr_canaux, quality, prev_dc;
    string mode;
    void init();
    Mat Y, Cb, Cr;

public:

    // Constructeurs
    Converter(Mat img, int taille_bloc);
    Converter(Mat img, int taille_bloc, int quality, int prev_dc, string mode);

    //  Processing
    void img_rgb2ycbcr();
    void sous_ech();
};

#endif // CONVERTER_HPP