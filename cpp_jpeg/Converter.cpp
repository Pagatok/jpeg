#include "Converter.hpp"
#include "Point3D.hpp"
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

const double PI = 3.14159265358979323846;

Mat getC(int taille_blocs){
    int N = taille_blocs;
    Mat C(N, N, CV_64F);
    float alpha = 1 / sqrt(N);
    for (int u = 0; u < N; u++){
        for (int x = 0; x < N; x++){
            C.at<float>(u, x) = alpha * cos((2 * x + 1) * u * PI / N / 2);
        }
        if (u == 0){
            alpha = sqrt(2 / N);
        }
    }
    return C;
}





// ----------------------------  CONSTRUCTEURS ----------------------------------

void Converter::init() {
    this->C = getC(taille_bloc);
    this->H = original_image.rows;
    this->W = original_image.cols;
    this->nbr_canaux = original_image.channels();
}

// Constructeur semi-complet
Converter::Converter(Mat img, int taille_bloc)
    : original_image(img), taille_bloc(taille_bloc), quality(0), prev_dc(0), mode("4:2:0"){
    init();
}

// Constructeur complet
Converter::Converter(Mat img, int taille_bloc, int quality, int prev_dc, string mode)
    : original_image(img), taille_bloc(taille_bloc), quality(50), prev_dc(0), mode("4:2:0"){
    init();
}


// ----------------------------  PROCESSING ----------------------------------

// conversion d'une image rgb en jpeg
void Converter::img_rgb2ycbcr(){

    vector<vector<Point3D>> new_image(H, vector<Point3D>(W));

    for(int i=0; i<H; i++){
        for(int j=0; j<W; j++){
            Vec3b pixel = (this->original_image).at<Vec3b>(i, j);
            Point3D ycbcr = pixel_rgb2ycbcr(pixel);
            new_image[i][j] = ycbcr;
        }
    }

    this->work_image = new_image;
}