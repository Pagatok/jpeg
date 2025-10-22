#include "Converter.hpp"
#include "Point3D.hpp"
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

const double PI = 3.14159265358979323846;

Mat getC(int taille_blocs){
    int N = taille_blocs;
    Mat C(N, N, CV_32F, Scalar(0.0f));
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

// conversion d'une image rgb en ycbcr
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

// Sous-echantillonage d'une image YCbCr
void Converter::sous_ech(){

    Mat Y(H, W, CV_32F, Scalar(0.0f));
    Mat Cb(H/2, W/2, CV_32F, Scalar(0.0f));
    Mat Cr(H/2, W/2, CV_32F, Scalar(0.0f));

    for(int i=0; i<(this->H)/2; i++){
        for(int j=0; j<(this->W)/2; j++){

            // Parcours du voisinage
            float sommeCb = 0;
            float sommeCr = 0;
            for(int k2=0; k2<2; k2++){
                for(int k1=0; k1<2; k1++){
                    Point3D pixel = work_image[2*i + k1][2*j + k2];
                    Y.at<float>(2*i+k1, 2*j+k2) = pixel.getX();
                    sommeCb += pixel.getY();
                    sommeCr += pixel.getZ();
                }
            }

            // Moyennage pour obtenir Cb et Cr
            Cb.at<float>(i, j) = sommeCb/4;
            Cr.at<float>(i, j) = sommeCr/4;
        }
    }

    this->Y = Y;
    this->Cb = Cb;
    this->Cr = Cr;
}



// for i, j in zip(range(H//2), range(W//2)):
            
//     bloc = self.image[k*i:k*(i+1), 2*j:2*(j+1), :]
//     avg_bloc = np.mean(bloc, axis=(0, 1))    # Moyenne par canal
    
//     new_crominances[i, j, :] = avg_bloc[1:]

// return new_crominances[:, :, 0], new_crominances[:, :, 1]