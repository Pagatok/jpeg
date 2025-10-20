import numpy as np
import sys
from PIL import Image



huff_table_dc_lumin = ["00", "010", "011", "100", '101', '110', '1110', '11110', '111110', '1111110', '11111110', '111111110']
huff_table_dc_chromin = ['00', '01', '10', '110', '1110', '11110', '111110', '1111110', '11111110', '111111110', '1111111110']

huffman_ac_luminance = {
    0x00: '1010',      # EOB
    0x01: '00',
    0x02: '01',
    0x03: '100',
    0x04: '1011',
    0x05: '11010',
    0x06: '1111000',
    0x07: '11111000',
    0x08: '1111110110',
    0x09: '1111111110000010',
    0x0F: '11111111001',  # ZRL (16 zéros)
    # ...
    # la table complète contient 162 entrées selon le standard JPEG
}

huffman_ac_chrominance = {
    0x00: '00',        # EOB
    0x01: '01',
    0x02: '100',
    0x03: '1010',
    0x04: '11000',
    0x05: '111000',
    0x06: '1111000',
    0x07: '111110100',
    0x08: '1111110110',
    0x09: '111111110100',
    0x0F: '1111111010', # ZRL
    # ...
    # la table complète contient 162 entrées
}


QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109,103, 77],
    [24, 35, 55, 64, 81, 104,113, 92],
    [49, 64, 78, 87,103, 121,120,101],
    [72, 92, 95, 98,112, 100,103, 99]
])

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def rgb2ycbcr_unit(R, G, B):
    Y = 16 + (65.738*R + 129.057*G + 25.064*B)/256
    Cb = 128 - (37.945*R + 74.494*G + 112.439*B)/256
    Cr = 128 + (112.439*R + 94.154*G + 18.285*B)/256
    
    return Y, Cb, Cr


def rgb_to_ycbcr(image):
    """
    image : numpy array de shape (H, W, 3) avec RGB en uint8
    return : numpy array de shape (H, W, 3) avec YCbCr en float32
    """
    # matrice de conversion (ITU-R BT.601)
    matrix = np.array([
        [ 0.299,   0.587,   0.114 ],
        [-0.1687, -0.3313,  0.5   ],
        [ 0.5,    -0.4187, -0.0813]
    ])

    shift = np.array([0, 128, 128])  # offset sur Cb et Cr

    # appliquer la conversion matricielle à tout le tableau
    ycbcr = image @ matrix.T + shift
    
    return ycbcr.astype(np.float32)


# Applique du padding en repetant la derniere ligne ou colonne plusieurs fois 
# pour avoir des dimensions multiple de multiple
def padding(image, multiple=8):
    pad_H, pad_W = [(multiple - (i % multiple)) % multiple for i in image.shape[0:2]]
        
    if image.ndim == 2:
        im_padded = np.pad(image, ((0, pad_H), (0, pad_W)), mode='edge')
    else:
        im_padded = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode='edge')
    
    return im_padded


def sousEch_chrom(image):

    H, W, C = image.shape
    
    if C < 2:
        sys.exit("Error: Expected number of canaux must be 2 or higher")
    
    new_crominances = np.zeros((H//2, W//2, 2))
    
    for i, j in zip(range(H//2), range(W//2)):
        
        bloc = image[i:i+2, j:j+2, :]
        avg_bloc = np.mean(bloc, axis=(0, 1))    # Moyenne par canal
        
        new_crominances[i, j, :] = avg_bloc[1:]

    return new_crominances


# quality_factor: int entre 1 et 100
def get_Qscaled(Q, quality_factor):

    if quality_factor < 1 or quality_factor > 100:
        sys.exit(f'Error: Quality factor should be an int between 1 and 100 (not {quality_factor})')
        
    if quality_factor == 50:
        return Q
        
    scale = 5000/quality_factor if quality_factor<50 else 200-(2*quality_factor)
    
    return np.floor((Q*scale + 50)/100)

# PArcours un bloc carre en zigzg et renvoie une liste
def parcoursZigzag(bloc):
    
    taille_bloc, W = bloc.shape
    
    if taille_bloc != W:
        sys.exit(f"Error: zigzg pathing need the bloc to be squared (not ({bloc.shape}))")
        return

    parcourus = []
    parite = -1
    
    # Parcours diagonale par diagonale
    for val_tot in range(0, 2*taille_bloc - 1, 1):
        
        # PArcours de la diagonale
        ligne = []
        for i in range(val_tot+1):
            j = val_tot - i
            if i < taille_bloc and j < taille_bloc:
                ligne.append(bloc[i, j])
            
        parcourus.extend(ligne[::parite])
        parite *= -1   
            
    return parcourus


# Renvoi la matrice de cosinus a appliquer pour faire la DCT
def getC(taille_bloc):
    
    x = np.arange(taille_bloc)
    u = x.reshape((taille_bloc,1))
    alpha = np.ones(taille_bloc) * np.sqrt(2/taille_bloc)
    alpha[0] = np.sqrt(1/taille_bloc)
    C = alpha[:,None] * np.cos((2*x+1) * u * np.pi / (2*taille_bloc))
    
    return C
    

# Parcours une suite de valeurs et la comvertit en une liste de couples de type (nbr0avantvalue, value)
def rlc(suite):
    
    ac = []
    count0 = 0

    for n in suite[1:]:
        
        if n == 0:
            count0 += 1
        else:
            ac.append((count0, n))
            count0 = 0
            
    if count0 != 0:
        ac.append("EOB")
    else:
        ac.append("EOB")
        
    return suite[0], ac
    
            
# Comvertit un entier en une chaine binaire (en inversant les bits pour les entiers negatifs)
def int_to_bin(n):
    
    if n < 0:
        string = int_to_bin(-n)
        new_string = ""
        for i in string:
            new_string += '1' if i == "0" else "1"   
        return new_string
    
    else:
        return "{0:b}".format(int(n))
        


# Encode un coeff DC en suivant les tables de Huffman
def huffmanDC(dc_diff, canal="luminance"):
    
    # on calcule son SIZE (nbr de bits pour l encoder sans le signe)
    magnitude = int_to_bin(dc_diff)
    size = len(magnitude)

    # code huffman pour SIZE (voir table huffman DC chrominance ou luminance)
    if canal == "luminance":
        huff_size = huff_table_dc_lumin[size]
    else:
        huff_size = huff_table_dc_chromin[size]

    return huff_size + magnitude


# Encode une liste AC encode en RLC en suivant les tables de Huffman
def huffmanAC(list_ac, canal="luminance"):
    
    huff_ac = ""
    
    for ac in list_ac:
        print(type(ac), ac)
        key = (ac[0] << 4) | ac[1].astype(int)
        
        if canal=="luminance":
            huff_ac += huffman_ac_luminance[key]
        else:
            canal="chrominance"
            huff_ac += huffman_ac_chrominance[key]
            
    return huff_ac
        





# decoupe une image single canal (H, W) en bloc puis applique la DCT sur chacun de ces blocs
def decoupe_dct_quantif(im_single_canal, Q, taille_bloc=8, canal="luminance"):
    
    # Padding de l image pour taille compatible
    image = padding(im_single_canal, multiple=taille_bloc)
    H, W = image.shape
    
    # Creation de la matrice de cosinus C
    C = getC(taille_bloc) 
    prev_dc = 0  
    first_dc = True

    # Parcours bloc par bloc et DCT+quantif sur chacun d eux    
    for i in range(H//taille_bloc):
        for j in range(W//taille_bloc):
        
            # Centrage des valeur du bloc
            bloc = image[i:i+taille_bloc, j:j+taille_bloc] - 128
            
            # Application de la DCT
            bloc_dct = C @ bloc @ C.T
            
            # Quantification
            bloc_quantif = np.floor(bloc_dct / Q)
            
            # Parcours en zigzag du bloc
            bloc_suite = parcoursZigzag(bloc_quantif)
            
            # Encodage RLC
            dc, ac = rlc(bloc_suite)
            
            # Encodage Huffman
            if first_dc:
                huff_dc = huffmanDC(dc, canal=canal)
                first_dc = False
            else:
                huff_dc = huffmanDC(dc - prev_dc, canal=canal)
            prev_dc = dc
            
            list_huff_ac = huffmanAC(ac, canal=canal)

        
    return 



class jpeg_convert:
    
    def __call__(self, img_path, quality=50, taille_bloc=8, mode="4:2:0"):
        
        # Initalisation
        self.image = np.array(Image.open(img_path))
        self.taille_bloc = taille_bloc
        self.quality = quality
        self.mode = mode
        self.H, self.W, self.nbr_canaux = (self.image).shape
        self.prev_dc = 0
        self.C = getC(self.taille_bloc)
        # Pipeline
        
        # COnversion de l 'image en YcbCr
        self.rgb_to_ycbcr()
        
        # Sous-echantillonage des chrominances
        y = self.image[:, :, 0]
        cb, cr = self.sousEch_chrom()
        
        # Decoupage en blocs
        self.blocs_y = self.decoupe_bloc(y)
        self.blocs_cb = self.decoupe_bloc(cb)
        self.blocs_cr = self.decoupe_bloc(cr)
        
        print(self.blocs_y.shape, self.blocs_cb.shape, self.blocs_cr.shape)


        # ======== Arrangement des codes de MCU ===============    
            
        flux_binaire_brut = ""
        
        # Parcours ligne par ligne puis colone par colone 
        for j in range(self.blocs_y.shape[1]//2):
            for i in range(self.blocs_y.shape[0]//2):
                
                # Codage de la MCU de coordonnes i, j
                flux_binaire_brut += self.code_mcu(i, j )
                
        print(flux_binaire_brut)
                
                
    # Trouve et encode les MCU        
    def code_mcu(self, i, j):
        
        if self.mode == "4:2:0":
            kh, kw = 2, 2
        elif self.mode == "4:2:2":
            kh, kw = 2, 1
        elif self.mode == "4:4:4":
            kh, kw == 1, 1
        else:
            sys.exit("Error: Mode sous-echantillonage non reconnu")
            
        flux_binaire = ""

        # Codage des Y
        ii, jj = 2*i, 2*j
        
        bloc1 = self.blocs_y[ii, jj, :, :]
        bloc2 = self.blocs_y[ii+1, jj, :, :]
        bloc3 = self.blocs_y[ii, jj+1, :, :]
        bloc4 = self.blocs_y[ii+1, jj+1, :, :]
        
        for bloc in [bloc1, bloc2, bloc3, bloc4]:
            flux_binaire += self.bloc2dcac(bloc, canal="luminance")
        
        
        # Codage des CbCR
        ii = kh*i
        jj = kw*j
        
        # Codage des Cb
        for kj in range(kw):
            for ki in range(kh):
                bloc = self.blocs_cb[ii+ki, jj+kj, :, :] 
                flux_binaire += self.bloc2dcac(bloc, canal="chrominance")
                
        # Codage des cr
        for kj in range(kw):
            for ki in range(kh):
                bloc = self.blocs_cr[ii+ki, jj+kj, :, :] 
                flux_binaire += self.bloc2dcac(bloc, canal="chrominance")
                
        return flux_binaire
    
            
        
        
    # Convertit self.images en YCbCr
    def rgb_to_ycbcr(self):
        """
        image : numpy array de shape (H, W, 3) avec RGB en uint8
        return : numpy array de shape (H, W, 3) avec YCbCr en float32
        """
        # matrice de conversion (ITU-R BT.601)
        matrix = np.array([
            [ 0.299,   0.587,   0.114 ],
            [-0.1687, -0.3313,  0.5   ],
            [ 0.5,    -0.4187, -0.0813]
        ])

        shift = np.array([0, 128, 128])  # offset sur Cb et Cr

        # appliquer la conversion matricielle à tout le tableau
        ycbcr = self.image @ matrix.T + shift
        
        self.image = ycbcr.astype(np.float32)


    # Sous-echantillone en h1,h2,h2 les chrominances d'une image
    # Renvoie les nouvelles bitmap des chrominances
    def sousEch_chrom(self):
        
        if self.mode == "4:4:4":
            return self.image[:, :, 1], self.image[:, :, 2]
        elif self.mode == "4:2:2":
            k = 1
        else:
            k = 2

        H, W, C = self.image.shape
        
        if C < 2:
            sys.exit("Error: Expected number of canaux must be 2 or higher")
        
        new_crominances = np.zeros((H//k, W//2, 2))

        
        for i, j in zip(range(H//2), range(W//2)):
            
            bloc = self.image[k*i:k*(i+1), 2*j:2*(j+1), :]
            avg_bloc = np.mean(bloc, axis=(0, 1))    # Moyenne par canal
            
            new_crominances[i, j, :] = avg_bloc[1:]

        return new_crominances[:, :, 0], new_crominances[:, :, 1]
    
    
    # Applique du padding en repetant la derniere ligne ou colonne plusieurs fois 
    # pour avoir des dimensions multiple de multiple
    def padding(image, multiple=8):
        pad_H, pad_W = [(multiple - (i % multiple)) % multiple for i in image.shape[0:2]]
            
        if image.ndim == 2:
            im_padded = np.pad(image, ((0, pad_H), (0, pad_W)), mode='edge')
        else:
            im_padded = np.pad(image, ((0, pad_H), (0, pad_W), (0, 0)), mode='edge')
        
        return im_padded

    
    # Decoupe une image single_canal en une liste de blocs carres de taille self.talle_bloc
    def decoupe_bloc(self, im_single_canal):
    
        # Padding de l image pour taille compatible
        image = padding(im_single_canal, multiple=self.taille_bloc)
        H, W = image.shape
        liste_blocs = np.zeros((H//self.taille_bloc, W//self.taille_bloc, 8, 8))
        
        for i in range(H//self.taille_bloc):
            for j in range(W//self.taille_bloc):
                bloc = image[i:i+self.taille_bloc, j:j+self.taille_bloc]
                liste_blocs[i, j, :, :] = bloc
                
        return liste_blocs
    
    
    # Renvoi une matrice Q de quantification adapate au facteur de qulite demande et du type luminance ou chrominance
    # quality_factor: int entre 1 et 100
    def get_Qscaled(self, canal="luminance"):
            
        Q = QY if canal=="luminance" else QC

        if self.quality < 1 or self.quality > 100:
            sys.exit(f'Error: Quality factor should be an int between 1 and 100 (not {self.quality})')
            
        if self.quality == 50:
            return Q
            
        scale = 5000/self.quality if self.quality<50 else 200-(2*self.quality)
        
        return np.floor((Q*scale + 50)/100)
    
    
    # Assemblage du flux binaire pour un groupe de blocs 
    # Exemple 4:2:0 -> 4Y, 1Cb, 1Cr
    def code_bloc(self, liste_blocsY, liste_blocsCbCr):
        
        bits = ""
        
        # Assemblage des blocs de luminance
        for bloc in liste_blocsY:
            dc, ac = self.bloc2dcac(bloc, self.C, canal="luminance")
            bits += dc
            bits += ac
            
        for bloc in liste_blocsCbCr:
            dc, ac = self.bloc2dcac(bloc, self.C, canal="chrominance")
            bits += dc
            bits += ac
            
        return bits
            
            
        
    
    
    # Renvoie le flux binaire encode avec Huffman d un bloc
    def bloc2dcac(self, bloc, canal="luminance"):
        
        # Centrage des valeur du bloc
        bloc = bloc - 128
        
        # Application de la DCT
        bloc_dct = self.C @ bloc @ self.C.T
        
        # Quantification
        bloc_quantif = np.floor(bloc_dct / self.get_Qscaled(canal=canal))
        
        # Parcours en zigzag du bloc
        bloc_suite = parcoursZigzag(bloc_quantif)
        
        # Encodage RLC
        dc, ac = rlc(bloc_suite)
    
        # Encodage Huffman de DC
        huff_dc = huffmanDC(dc - self.prev_dc, canal=canal)
        self.prev_dc = dc - self.prev_dc
        
        # Encodage Huffman de AC
        list_huff_ac = huffmanAC(ac, canal=canal)
        
        return huff_dc + list_huff_ac
        
        







class Tests():
    
    def __call__(self):
        self.test_parcoursZigzag()
        self.test_rlc()
        
    
    def test_parcoursZigzag(self):
        
        # Matrice de test
        Z = np.array([
            [1,  2,  6,  7,  15, 16, 28, 29],
            [3,  5,  8,  14, 17, 27, 30, 43],
            [4,  9,  13, 18, 26, 31, 42, 44],
            [10, 12, 19, 25, 32, 41, 45, 54],
            [11, 20, 24, 33, 40, 46, 53, 55],
            [21, 23, 34, 39, 47, 52, 56, 61],
            [22, 35, 38, 48, 51, 57, 60, 62],
            [36, 37, 49, 50, 58, 59, 63, 64]
        ])
        
        Z_p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
               39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 
               57, 58, 59, 60, 61, 62, 63, 64]
        
        assert parcoursZigzag(Z) == Z_p
        
        
    def test_rlc(self):
        
        Z1 = [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        
        
        Z_f1 = [(2, 1), (3, 2), (3, 1), (6, 1), (4, 2), (4, 1), (7, 1), (7, 2), (3, 1), (6, 1), "EOB"]

        assert rlc(Z1) == (0, Z_f1)
        
        
        
if __name__ == "__main__":
    converter = jpeg_convert()
    converter("sample_1280x853.bmp")