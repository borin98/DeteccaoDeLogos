import cv2
import numpy as np
from os.path import isfile, join
from os import listdir

def boundingBox(imagem, deteccoes, i=0, j = 1):
    """
    Função que faz a detecção do objeto de
    interesse e o coloca em um retângulo

    :return:
    """

    for (x, y, l, a) in deteccoes:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

    cv2.imshow("Detector"+str(j)+" de logo "+str( i+1 ), imagem )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

def carregaImagens ( ) :
    """
    Função que monta o vetor de imagens presentes
    em uma pasta

    :return:
    """

    # carregando as imagens

    path = "Logo/Teste"
    apenasFiles = [f for f in listdir(path) if isfile(join(path, f))]
    print(apenasFiles)
    images = np.empty(len(apenasFiles), dtype=object)

    for n in range(0, len(apenasFiles)):
        images[n] = cv2.imread(join(path, apenasFiles[n]))

    return images


def main ( ) :

    imagens = carregaImagens()
    Classificador = cv2.CascadeClassifier("Logo/negativas/Classificador/cascade.xml")

    for i in range ( len ( imagens ) ) :
        print("\nImagem {}\n".format(i + 1))

        imagem = imagens[i]

        imagemConvertida = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # fazendo a classificação dos dados
        deteccoes = Classificador.detectMultiScale(imagemConvertida,
                                                    scaleFactor = 1.33,
                                                    minNeighbors = 10,
                                                    minSize = ( 40, 40) )

        boundingBox ( imagem, deteccoes, i, 1 )

if __name__ == '__main__':
    main()