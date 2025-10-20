from PIL import Image
import numpy as np
import sys
from convert_tools import *
from matplotlib import pyplot as plt


im = np.array(Image.open("sample_1280x853.bmp"))
H, W, C = im.shape


# ========== Conversion en YCbCr ============

im_converted = rgb_to_ycbcr(im)


# ========== Sous-Echantillonage ============

# Padding en nombre pair de colonnes et de lignes pour la sous-echantillonage en 4:2:0
im_converted = padding(im_converted, multiple=2)

# Sous_echantillonage en 4:2:0 des chrominances
chrominances = sousEch_chrom(im_converted)
y = im_converted[:, :, 0]
cb = chrominances[:, :, 0]
cr = chrominances[:, :, 1]

# print(y.shape, [x/8 for x in y.shape])
# print(cb.shape, [x/8 for x in cb.shape])
# print(cr.shape, [x/8 for x in cr.shape])





# ======= Decoupage en blocs et DCT =========


# Creation de la matrice de quantification standart
quality_factor = 50
QY = get_Qscaled(np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109,103, 77],
    [24, 35, 55, 64, 81, 104,113, 92],
    [49, 64, 78, 87,103, 121,120,101],
    [72, 92, 95, 98,112, 100,103, 99]
]), quality_factor)

QC = get_Qscaled(np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]), quality_factor)



list_huff_dc, list_ac = decoupe_dct_quantif(y, QY, canal="luminance")
list_huff_dc, list_ac = decoupe_dct_quantif(cb, QC, canal="chrominance")
list_huff_dc, list_ac = decoupe_dct_quantif(cr, QC, canal="chrominance")


print("All right")
sys.exit()
        
    
# Pour faire les tests du package
Tests()()
sys.exit()

