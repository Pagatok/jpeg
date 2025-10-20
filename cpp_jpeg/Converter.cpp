#include "Converter.hpp"
#include <cmath>
#include <iostream>

// # Renvoi la matrice de cosinus a appliquer pour faire la DCT
// def getC(taille_bloc):

//     x = np.arange(taille_bloc)
//     u = x.reshape((taille_bloc,1))
//     alpha = np.ones(taille_bloc) * np.sqrt(2/taille_bloc)
//     alpha[0] = np.sqrt(1/taille_bloc)
//     C = alpha[:,None] * np.cos((2*x+1) * u * np.pi / (2*taille_bloc))

//     return C
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

// Constructeur semi-complet
Converter::Converter(Mat *img, int taille_bloc, int H, int W, int nbr_canaux)
    : image(img), taille_bloc(taille_bloc), H(H), W(W), nbr_canaux(nbr_canaux),
      quality(0), prev_dc(0), mode("default")
{
    // éventuellement initialiser C = getC(taille_bloc);
}

// Constructeur complet
Converter::Converter(Mat *img, int taille_bloc, int H, int W, int nbr_canaux, int quality,
                     int prev_dc, const string &mode)
    : image(img), taille_bloc(taille_bloc), H(H), W(W), nbr_canaux(nbr_canaux),
      quality(quality), prev_dc(prev_dc), mode(mode)
{
    // éventuellement initialiser C = getC(taille_bloc);
}