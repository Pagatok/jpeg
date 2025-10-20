#include <iostream>
#include <string>
#include <opencv2/opencv.hpp> // NOLINT
#include "Point3D.hpp"
#include "Converter.hpp"

using namespace cv;
using namespace std;

string huff_table_dc_lumin[12] = {"00", "010", "011", "100", "101", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110"};
string huff_table_dc_chromin[11] = {"00", "01", "10", "110", "1110", "11110", "111110", "1111110", "11111110", "111111110", "1111111110"};

int main()
{

    Point3D p(126, 234, 66);
    Point3D p2 = rgb2ycbcr(p);
    p2.afficher();

    // Manip image basique
    Mat img = imread("../sample_1280x853.bmp"); // lire l'image
    if (img.empty())
    {
        cout << "Erreur lors de la lecture de l'image" << endl;
        return -1;
    }

    imshow("Image", img); // afficher
    waitKey(0);           // attendre touche
    return 0;
}