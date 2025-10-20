#include "Converter.hpp"
#include <cmath>
#include <iostream>


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
    this->H = image.rows;
    this->W = image.cols;
    this->nbr_canaux = image.channels();
}

// Constructeur semi-complet
Converter::Converter(Mat img, int taille_bloc)
    : image(img), taille_bloc(taille_bloc), quality(0), prev_dc(0), mode("4:2:0"){
    init();
}

// Constructeur complet
Converter::Converter(Mat img, int taille_bloc, int quality, int prev_dc, string mode)
    : image(img), taille_bloc(taille_bloc), quality(0), prev_dc(0), mode("4:2:0"){
    init();
}


// ----------------------------  PROCESSING ----------------------------------