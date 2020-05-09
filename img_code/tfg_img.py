# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
from kneed import KneeLocator
from tabulate import tabulate

###########################################
###   LECTURA E IMPRESIÓN DE IMÁGENES   ###
###########################################

""" Lee una imagen ya sea en grises o en color. Devuelve la imagen.
- file_name: archivo de la imagen.
- flag_color (op): modo en el que se va a leer la imagen -> grises o color. Por defecto será en color.
"""
def read_img(file_name, flag_color = 1):
    if flag_color == 0:
        print("Leyendo '" + file_name + "' en gris.")
    elif flag_color==1:
        print("Leyendo '" + file_name + "' en color.")
    else:
        print("flag_color debe ser 0 o 1")

    img = cv2.imread(file_name, flag_color)

    if flag_color==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    return img

""" Normaliza una matriz.
- image: matriz a normalizar.
- img_title (op): título de la imagen. Por defecto ' '.
"""
def normaliza(image, img_title = " "):
    norm = np.copy(image)
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max>255 or min<0:
            print("Normalizando imagen '" + img_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    norm[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = [np.amax(image[:,:,0]), np.amax(image[:,:,1]), np.amax(image[:,:,2])]
        min = [np.amin(image[:,:,0]), np.amin(image[:,:,1]), np.amin(image[:,:,2])]

        if max[0]>255 or max[1]>255 or max[2]>255 or min[0]<0 or min[1]<0 or min[2]<0:
            print("Normalizando imagen '" + img_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        norm[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

    return norm

""" Imprime una imagen a través de una matriz.
- image: imagen a imprimir.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- img_title(op): título de la imagen. Por defecto 'Imagen'
- window_title (op): título de la ventana. Por defecto 'Ejercicio'
"""
def show_img(image, flag_color=1, img_title = "Imagen", window_title = "Ejercicio"):
    img_show = normaliza(image, img_title)            # Normalizamos la matriz
    img_show = img_show.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(window_title) # Ponemos nombre a la ventana
    if flag_color == 0:
        plt.imshow(img_show, cmap = "gray")
    else:
        plt.imshow(img_show)
    plt.title(img_title)              # Ponemos nombre a la imagen
    plt.xticks([])                      # Se le pasa una lista de posiciones en las que se deben colocar los
    plt.yticks([])                      # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()

""" Lee una lista de imágenes ya sea en grises o en color. Devuelve la lista de imágenes leída.
- image_list: lista de imágenes a concatenar.
- flag_color (op): modo en el que se van a leer las imágenes. Por defecto en color.
"""
def read_img_list(file_name_list, flag_color = 1):
    image_list = []

    for i in file_name_list:
        img = read_img(i, flag_color)
        image_list.append(img)

    return image_list

""" Muestra múltiples imágenes en una ventena Matplotlib.
- img_list: La lista de imágenes.
- title_list: Lista de títulos de las imágenes.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- window_title (op): título de la ventana. Por defecto 'Imágenes con títulos'
"""
def show_img_list(img_list, title_list, flag_color=0, window_title="Haar Wavelets"):
    fig = plt.figure(figsize=(9, 9))
    fig.canvas.set_window_title(window_title)
    img_show_list = []

    for i in range(len(img_list)):
        img_show_list.append(normaliza(img_list[i], title_list[i]).astype(np.uint8))

    for i, a in enumerate(img_show_list):
        display = fig.add_subplot(2, 2, i + 1) #(1, 4, i + 1)
        if flag_color == 0:
            display.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        else:
            display.imshow(a, interpolation="nearest")
        display.set_title(title_list[i], fontsize=10)
        display.set_xticks([])          # Deshabilitamos xticks
        display.set_yticks([])          # e y_ticks

    fig.tight_layout()
    plt.show()

""" Guarda la imagen en el file_name proporcionado.
- file_name: archivo donde guardar la imagen.
- img: imagen a guardar.
- normalize: indica si normalizar la imagen antes de guardarla.
"""
def save_img(file_name, img, normalize=False):
    print("Guardando imagen '" + file_name + "'.")
    if(normalize):
        im = normaliza(img, file_name)
    else:
        im = img.copy()
    im=np.array(im, dtype=np.uint8)
    if(len(im.shape)==3):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_name, im)

##########################
###   HAAR TRANSFORM   ###
##########################

""" Realiza el algoritmo Haar Wavelet de manera recursiva realizando las particiones
adecuadas y calculando sus medias y diferencias. Devuelve una lista con los coeficientes.
- list: lista a la que hacer el split de haar.
- offset: parte fija del split.
"""
def haar_transform(list, offset):
    if(len(list) >= 2):
        avgs = []
        difs = []
        for i in range(0, len(list), 2):
            avgs.append((list[i] + list[i+1]) / math.sqrt(2))
            difs.append((list[i] - list[i+1]) / math.sqrt(2))
        return haar_transform(avgs, difs + offset)

    else:
        return list + offset

""" Calcula la transformada de Haar por filas de una imágen.
Devuelve los coeficientes de Haar después de aplicar el proceso por filas.
- img: imagen original a transformar.
"""
def haar_row(img):
	row = []
	for pixels in img:
		row.append(haar_transform(pixels, []))
	return row

""" Calcula la transformada de Haar una imágen.
Devuelve la imagen de los coeficientes de Haar.
- img: imagen original a transformar.
- img_title(op): título de la imagen. Por defecto "".
"""
def haar_image(img, img_title=""):
    if(img_title != ""):
        print("Calculando la transformada de Haar a la imagen '" + img_title +"'.")

    if(len(img.shape)==2):
        by_row = haar_row(img)              # por filas
        haar_img = zip(*by_row)             # transponemos
        haar_img = haar_row(haar_img)       # por columnas
        haar_img = np.array(haar_img)
        haar_img = np.transpose(haar_img)   # transponemos
    else:
        haar_img = np.zeros(img.shape)
        for k in range(3):
            haar_img[:,:,k] = haar_image(img[:,:,k], "")

    return haar_img

""" Calcula la transformada inversa de Haar. Devuelve la lista resultante.
- front: primera parte de la lista.
- the_rest: segunda parte de la lista.
- power: 2 elevado a este exponente me dice el índice de the_rest en la lista.
"""
def reverse_haar(front, the_rest, power):
	reverse = []

	for i in range(len(front)):
		reverse.append((front[i] + the_rest[i]) / math.sqrt(2))
		reverse.append((front[i] - the_rest[i]) / math.sqrt(2))

	if(len(the_rest) > len(reverse)):
		the_rest = the_rest[2**power:]
		power += 1
		return reverse_haar(reverse, the_rest, power)
	else:
		return reverse

""" Dada una transformada de Haar de una imagen calcula su inversa por filas.
Devuelve la inversa por filas.
- haar_img: imagen después de la transformada de Haar.
- img_title(op): título de la imagen. Por defecto "".
"""
def reverse_row(haar_img, img_title=""):
    row = []
    for pixels in haar_img:
        row.append(reverse_haar(pixels[:1], pixels[1:], 0))
    return row

""" Dada una transformada de Haar de una imagen calcula su inversa.
Devuelve la imagen original.
- haar_img: imagen después de la transformada de Haar.
- img_title(op): título de la imagen. Por defecto "".
"""
def reverse_image(haar_img, img_title=""):
    if(img_title != ""):
        print("Restaurando la imagen original de la transformada de Haar de '" + img_title + "'.")

    if(len(haar_img.shape)==2):
        by_row = reverse_row(haar_img)      # por filas
        rev_haar = zip(*by_row)             # transponemos
        rev_haar = reverse_row(rev_haar)    # por columnas
        rev_haar = np.array(rev_haar)
        rev_haar = np.transpose(rev_haar)   # tranponemos

        #rev_haar = rev_haar.astype(np.uint8)
        #rev_haar = rev_haar.astype(np.float64)
    else:
        rev_haar = np.zeros(haar_img.shape)
        for k in range(3):
            rev_haar[:,:,k] = reverse_image(haar_img[:,:,k], "")

    return rev_haar

""" Cuenta el número de elementos distintos de cero de una imagen
Devuelve la imagen después de aplicar el thresholding.
- img: imagen sobre la que calcular el número de elementos distintos de cero.
- rows: filas para el conteo.
- cols: columnas para el conteo.
"""
def not_zero(img, rows, cols):
    cont = 0

    if(len(img.shape)==2):
        for i in range(rows):
            for j in range(cols):
                if(img[i][j]!=0):
                    cont += 1
    else:
        for k in range(3):
            cont += not_zero(img[:,:,k], rows, cols)

    return cont

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el thresholding.
- haar_img: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- img_title(op): título de la imagen. Por defecto "".
"""
def thresholding(haar_img, epsilon, img_title=""):
    if(img_title != ""):
        print("Aplicando thresholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, img_title))
    threshold_img = haar_img.copy()

    if(len(haar_img.shape)==2):
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if (abs(haar_img[i][j]) < epsilon):
                    threshold_img[i][j] = 0.0

    else:
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                for k in range(len(haar_img[0][0])):
                    if (abs(haar_img[i][j][k]) < epsilon):
                        threshold_img[i][j][k] = 0.0

    return threshold_img

""" Se queda con la mejor aproximación de m-términos.
Devuelve la imagen después de aplicar el algoritmo.
- haar_img: imagen después de la transformada de Haar.
- m: número de términos que nos vamos a quedar.
- img_title(op): título de la imagen. Por defecto "".
"""
def m_term(haar_img, m, img_title=""):
    if(img_title != ""):
        print("Aplicando algoritmo de m-términos (m={}) a la transformada de Haar de '{}'."
              .format(m, img_title))

    if(len(haar_img.shape)==2):
        list = np.zeros(m)

        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if(m>0):
                    list[i+j] = abs(haar_img[i][j])
                    m -= 1
                else:
                    val = abs(haar_img[i][j])
                    if(np.amin(list) < val):
                        list[list==np.amin(list)] = val

        m_term_img = thresholding(haar_img, np.amin(list), "")


    else:
        m_term_img = np.zeros(haar_img.shape)
        for k in range(3):
            m_term_img[:,:,k] = m_term(haar_img[:,:,k], m, "")

    return m_term_img

""" Calcula el error medio de la imagen original y su aproximación.
Devuelve el error medio.
- img: imagen original.
- back_img: imagen aproximada.
- img_title(op): título de la imagen. Por defecto "".
"""
def error(img, back_img, img_title=""):
    err = 0

    if(len(img.shape)==2):
        for i in range(len(img)):
            for j in range(len(img[0])):
                err += abs(img[i][j]-back_img[i][j])
        err = err / (len(img)*len(img[0]))
    else:
        for i in range(len(img)):
            for j in range(len(img[0])):
                for k in range(len(img[0][0])):
                    err += abs(img[i][j][k]-back_img[i][j][k])
        err = err / (len(img)*len(img[0])*len(img[0][0]))

    err = round(err, 4)
    if(img_title != ""):
        print("Error medio de '{}' y su aproximación: {}.".format(img_title, err))

    return err

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- img: imagen a recortar.
- sq (op): indica si extender de manera cuadrada. Por defecto False.
- img_title(op): título de la imagen. Por defecto "".
"""
def crop_img(img, sq=False, img_title=""):
    p = 2**int(math.log(len(img), 2))
    q = 2**int(math.log(len(img[0]), 2))

    if(sq):     # ajustamos p y q en caso de ser cuadrada
        if(p<q): q=p
        else: p=q

    if(img_title != ""):
        if(len(img.shape)==3):
            print("Recortando imagen '{}' a tamaño ({}, {}, 3).".format(img_title, p, q))
        else:
            print("Recortando imagen '{}' a tamaño ({}, {}).".format(img_title, p, q))

    a = int((len(img)-p)/2)
    b = int((len(img[0])-q)/2)

    return img[a:(a+p), b:(b+q)]

""" Extiende a la imagen de tamaño 2^p*2^q más pequeña dentro de la que cabe.
Devuelve la imagen extendida.
- img: imagen a extender.
- sq (op): indica si extender de manera cuadrada. Por defecto False.
- img_title(op): título de la imagen. Por defecto "".
"""
def extend_img(img, sq=False, img_title=""):
    n = math.log(len(img), 2)
    m = math.log(len(img[0]), 2)
    to_extend = False

    if(int(n)<n):
        n = int(n) + 1
        to_extend = True
    else:
        n = int(n)
    if(int(m)<m):
        m = int(m) + 1
        to_extend = True
    else:
        m = int(m)

    if(to_extend):
        p = 2**n
        q = 2**m

        if(sq):     # ajustamos p y q en caso de ser cuadrada
            if(p>q): q=p
            else: p=q

        if(len(img.shape)==3):
            if(img_title != ""):
                print("Extendiendo imagen '{}' a tamaño ({}, {}, 3).".format(img_title, p, q))
            ext = np.zeros((p, q, 3), dtype=np.float64)
        else:
            if(img_title != ""):
                print("Extendiendo imagen '{}' a tamaño ({}, {}).".format(img_title, p, q))
            ext = np.zeros((p, q), dtype=np.float64)

        for i in range(len(img)):
            for j in range(len(img[0])):
                ext[i][j] = img[i][j]

        return ext
    else:
        return img

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- img: imagen a recortar.
- rows: número de filas a recortar.
- cols: número de columnas a recortar.
- img_title(op): título de la imagen. Por defecto "".
"""
def crop_size(img, rows, cols, img_title=""):
    if(img_title != ""):
        if(len(img.shape)==2):
            print("Recortando imagen '{}' a tamaño ({}, {}).".format(img_title, rows, cols))
        else:
            print("Recortando imagen '{}' a tamaño ({}, {}, {}).".format(img_title, rows, cols, 3))
    return img[:rows, :cols]

""" Imprime por pantalla el tamaño en bytes de los archivos y el factor de compresión.
Devuelve el factor de compresión.
- file_org: archivo original.
- file_rev: arhivo greedy.
"""
def diff_size(img, comp_img):
    comp_factor = img.nbytes / np.array(comp_img).nbytes
    print("El archivo original pesa {} bytes y la compresión del greedy {} bytes."
          .format(img.nbytes, np.array(comp_img).nbytes))
    print("Factor de compresión {}".format(comp_factor))
    return comp_factor

""" Concatena dos imágenes de la manera más adecuada. Devuelve la imagen resultante.
- img1: imagen 1 a concatenar.
- img2: imagen 2 a concatenar.
"""
def concat(img1, img2):
    if(img1.shape[0] >= img1.shape[1]):
        if(len(img1.shape)==3):
            concat = np.zeros((img1.shape[0], 2*img1.shape[1], img1.shape[2]))
        else:
            concat = np.zeros((img1.shape[0], 2*img1.shape[1]))

        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                concat[i][j] = img1[i][j]
                concat[i][j+img1.shape[1]] = img2[i][j]
    else:
        if(len(img1.shape)==3):
            concat = np.zeros((2*img1.shape[0], img1.shape[1], img1.shape[2]))
        else:
            concat = np.zeros((2*img1.shape[0], img1.shape[1]))

        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                concat[i][j] = img1[i][j]
                concat[i+img1.shape[0]][j] = img2[i][j]

    return concat

""" Calcula la norma de la función de Haar hn. Devuelve su norma.
- n: índice de la función de Haar.
"""
def norm_hn(n):
    if(n==1):
        return 1.0
    # Calculamos k
    l = math.log(n,2);
    k = int(l);
    if l - k == 0:
        k = k-1
    return 1/math.sqrt(2**k)

""" Calcula la norma de la función de Haar en un píxel. Devuelve su norma.
- row: fila del píxel.
- col: columna del píxel.
"""
def norm_pixel(row, col):
    return norm_hn(row+1) * norm_hn(col+1)

""" Experimento greedy a realizar.
Devuelve la compresión, porcentaje de descarte, ratio de dispersión,
error medio y factor de compresión.
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
- fun: función de aproximación (thresholding, m_term).
- param: parámetro de la función de aproximación.
- print_mat (op): indica si se deben imprimir las matrices. Por defecto 'False'.
- show_im (op): indica si mostrar las imágenes. Por defeto 'True'.
- save_im (op): indica si guardar las imágenes. Por defecto 'True'.
"""
def experiment(img_title, flag, fun, param, print_mat=False, show_im=True, save_im=True):
    print("\n###############################################")
    print("\tTranformada de Haar de {}".format(img_title))
    print("###############################################\n  ")
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    ext = extend_img(img, False, img_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, img_title)
    # Aplicándole el algoritmo greedy
    not_zero_before = not_zero(haar_img, len(img), len(img[0]))
    greedy_img = fun(haar_img, param, img_title)
    not_zero_after = not_zero(greedy_img, len(img), len(img[0]))
    # Calulando ratio y perc
    if(len(img.shape)==2):
        total = len(img)*len(img[0])
    else:
        total = len(img)*len(img[0])*len(img[0][0])
    perc = round(100*(total-not_zero_after)/total, 2)
    if(not_zero_after!=0):
        ratio = round(not_zero_before/not_zero_after, 4)
    else:
        ratio = math.inf
    if(img_title != ""):
        print("Número de píxeles anulados: {} ({}%).".format(total-not_zero_after, perc))
        print("Ratio de dispersión: {}.".format(ratio))
    # Comprimimos
    comp_img, cent = compress_img(greedy_img, img_title)

    """
    AQUÍ REALIZARÍAMOS EL ENVÍO DE 'comp_img'
    """

    # Descomprimimos
    uncomp_img = uncompress_img(comp_img, cent, img_title)
    # Restaurando la imagen original
    rev_img = reverse_image(uncomp_img, img_title)

    if(rev_img.shape != img.shape): # recorta si hemos extendido
        rev_img = crop_size(rev_img, img.shape[0], img.shape[1], img_title)

    if (print_mat):  # si queremos imprimir las matrices
        print("\nMatriz de la imagen original:")
        print(img)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_img)
        print("\nMatriz después del algoritmo greedy:")
        print(greedy_img)
        print("\nMatriz de la descompresión:")
        print(uncomp_img)
        print("\nMatriz de la imagen restaurada:")
        print(rev_img)
        print()

    # Calulamos el error medio de la imagen original y la revertida
    err = error(img, rev_img, img_title)

    # Mostramos diferentes imágenes
    #show_img_list([img, haar_img, greedy_img, rev_img],
    #            ["Original", "2D Haar Transform", "Greedy", "Return to Original"])

    # Concatenamos la imagen original y la recuperada para pintarla
    concat_img = concat(img, normaliza(rev_img, img_title))
    if(show_im):
        show_img(concat_img, flag, img_title, "Haar wavelets")

    if(save_im):    # Guardamos las imágenes
        save_img("results/rev" + str(param) + "_" + img_title, rev_img, True)
        save_img("results/greedy" + str(param) + "_" + img_title, greedy_img, False)
        save_img("results/concat" + str(param) + "_" + img_title, concat_img, False)

    factor = diff_size(img, comp_img)

    return comp_img, perc, ratio, err, factor

""" Obtenemos el gradiente de la imagen
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
"""
def getDerivates(img_title, flag):
    print("\n###############################################")
    print("\tDerivada de '{}'".format(img_title))
    print("###############################################\n  ")
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    ext = extend_img(img, False, img_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, img_title)

    der = haar_img[len(img)//2:, len(img[0])//2:]

    max = np.amax(der)
    min = np.amin(der)
    for i in range(der.shape[0]):
        for j in range(der.shape[1]):
            der[i][j] = (der[i][j]-min)/(max-min) * 255

    show_img(der, 0, img_title, "Derivadas")
    save_img("results/der_" + img_title, der)

    return der

###########################
###   COMPRESSION RLE   ###
###########################

""" Comprime la imagen greedy_img.
Devuelve la compresión como lista y el valor centinela.
- greedy_img: imagen de los coeficientes greedy.
- img_title(op): título de la imagen. Por defecto "".
"""
def compress_img(greedy_img, img_title=""):
    comp = []

    if(img_title != ""):
        print("Comprimiendo imagen de coeficientes de '{}'.".format(img_title))

    if(len(greedy_img.shape)==2):
        cent = int(np.amax(greedy_img)) + 1

        for i in range(len(greedy_img)):
            row = []
            count = 0

            for j in range(len(greedy_img[0])-1):
                if(greedy_img[i][j]==0):
                    count +=1
                elif(count>0):
                    row.append(cent)
                    row.append(count)
                    count = 0
                    row.append(greedy_img[i][j])
                else:
                    row.append(greedy_img[i][j])

            if(count>0):
                if(greedy_img[i][-1]==0):
                    row.append(cent)
                    row.append(count+1)
                else:
                    row.append(cent)
                    row.append(count)
                    row.append(greedy_img[i][-1])
            else:
                row.append(greedy_img[i][-1])

            comp.append(row)

    else:
        cent = []

        for k in range(len(greedy_img[0][0])):
            co, ce = compress_img(greedy_img[:,:,k], "")
            comp.append(co)
            cent.append(ce)

    return comp, cent

""" Descomprime una imagen comprimida. Devuelve la imagen.
- lists: compresión.
- cent: valor centinela.
- img_title(op): título de la imagen. Por defecto "".
"""
def uncompress_img(lists, cent, img_title=""):
    if(img_title != ""):
        print("Descomprimiendo imagen '{}'.".format(img_title))

    if(isinstance(cent, list) == False):    # B/N
        un = []
        act = False

        for li in lists:
            row = []
            for j in range(len(li)):
                if(act == True):
                    act = False
                    for i in range(li[j]):
                        row.append(0)
                elif(li[j]!=cent):
                    row.append(li[j])
                else:   # li[j]==cent
                    act = True

            un.append(row)
        img = np.array(un)

    else:   # Color
        for k in range(len(cent)):
            un = uncompress_img(lists[k], cent[k], "")
            if(k==0):
                img = np.empty((un.shape[0], un.shape[1], 3))
            img[:,:,k] = un

    return img

########################
###   OPTIMIZATION   ###
########################

""" Experimento greedy a realizar.
Devuelve el error, porcentaje de descarte y ratio de descompresión obtenidos.
- img: imagen inicial sobre la que realizar el experimento.
- thr: parámetro de la función de aproximación.
"""
def experiment_opt(img, thr):
    ext = extend_img(img, False, "") # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, "")
    # Aplicándole el algoritmo greedy
    not_zero_before = not_zero(haar_img, len(img), len(img[0]))
    greedy_img = thresholding(haar_img, thr, "")
    not_zero_after = not_zero(greedy_img, len(img), len(img[0]))
    # Calulando ratio y perc
    if(len(img.shape)==2):
        total = len(img)*len(img[0])
    else:
        total = len(img)*len(img[0])*len(img[0][0])
    perc = round(100*(total-not_zero_after)/total, 2)
    if(not_zero_after!=0):
        ratio = round(not_zero_before/not_zero_after, 4)
    else:
        ratio = math.inf
    # Restaurando la imagen original
    rev_img = reverse_image(greedy_img, "")

    if(rev_img.shape != img.shape): # recorta si hemos extendido
        rev_img = crop_size(rev_img, img.shape[0], img.shape[1], "")

    # Calulamos el error medio de la imagen original y la revertida
    err = error(img, rev_img, "")
    return err, perc, ratio

""" Optimización del error medio. Devuelve el punto 'knee' y el umbral y error asociado.
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
"""
def optimization(img_title, flag):
    print("\n###############################################")
    print("\tOptimizando umbral de '{}'".format(img_title))
    print("###############################################\n  ")
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    thrs = []; errs = []; pers = []; rats = []

    for thr in range(1,20,1):
        err, per, rat = experiment_opt(img, thr)
        thrs.append(thr); errs.append(err); pers.append(per); rats.append(rat)
    for thr in range(20,40,2):
        err, per, rat = experiment_opt(img, thr)
        thrs.append(thr); errs.append(err); pers.append(per); rats.append(rat)

    # Imprimo las listas
    print("Umbrales:")
    print(thrs)
    print("Errores:")
    print(errs)
    print("Porcentajes de descarte:")
    print(pers)
    print("Ratios de dispersión:")
    print(rats)

    # Calculo el 'knee'
    kneedle = KneeLocator(pers, errs, S=1.0, curve='convex', direction='increasing')
    # Busco el umbral asociado a ese 'knee'
    for i in range(len(pers)):
        if (pers[i] == kneedle.knee):
            opt_thr = thrs[i]
            opt_err = errs[i]

    print("El punto 'knee' es: {}".format(round(kneedle.knee, 2)))
    print("El umbral asociado es: {}".format(opt_thr))

    # Imprimo las gráficas
    plt.plot(pers, errs, '-o', linewidth=1)
    plt.vlines(kneedle.knee, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Punto 'knee'")
    plt.xlabel("Porcentaje de descartados")
    plt.ylabel("Error medio")
    plt.legend()
    plt.title("Relación porcentaje de descarte - error para '{}'".format(img_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("results/graf_pers_" + img_title)
    plt.show()

    plt.plot(thrs, errs, '-o', linewidth=1)
    plt.vlines(opt_thr, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Umbral del punto 'knee'")
    plt.xlabel("Umbral")
    plt.ylabel("Error medio")
    plt.legend(loc="lower right")
    plt.title("Relación umbral - error para '{}'".format(img_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("results/graf_thrs_" + img_title)
    plt.show()

    return opt_thr, kneedle.knee, opt_err

#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    N = 8
    list = [['Ejemplo', 'Umbral', 'Descartes (%)', 'Ratio dispersión', 'Error medio', 'Factor de compresión'],
         ['Lena', 3.0],
         ['León', 3.0],
         ['Lena', 40.0],
         ['León', 50.0],
         ['Lena (color)', 3.0],
         ['Alhambra', 3.0],
         ['Lena (color)', 40.0],
         ['Alhambra', 40.0]]

    per = np.zeros(N); rat = np.zeros(N); err = np.zeros(N); fac = np.zeros(N)

    _, per[0], rat[0], err[0], fac[0] = experiment("lena.png", 0, thresholding, 3.0, show_im=False, save_im=False)
    _, per[1], rat[1], err[1], fac[1] = experiment("lion.png", 0, thresholding, 3.0, show_im=False, save_im=False)
    _, per[2], rat[2], err[2], fac[2] = experiment("lena.png", 0, thresholding, 40.0, show_im=False, save_im=False)
    _, per[3], rat[3], err[3], fac[3] = experiment("lion.png", 0, thresholding, 40.0, show_im=False, save_im=False)
    _, per[4], rat[4], err[4], fac[4] = experiment("lena_color.png", 1, thresholding, 3.0, show_im=False, save_im=False)
    _, per[5], rat[5], err[5], fac[5] = experiment("alham.png", 1, thresholding, 3.0, show_im=False, save_im=False)
    _, per[6], rat[6], err[6], fac[6] = experiment("lena_color.png", 1, thresholding, 40.0, show_im=False, save_im=False)
    _, per[7], rat[7], err[7], fac[7] = experiment("alham.png", 1, thresholding, 40.0, show_im=False, save_im=False)

    for k in range(1,N+1):
        list[k].append(per[k-1])
        list[k].append(rat[k-1])
        list[k].append(err[k-1])
        list[k].append(fac[k-1])

    print()
    print(tabulate(list, headers='firstrow', tablefmt='fancy_grid'))

    getDerivates("lena.png", 0)

    optimization("lena.png", 0)
    optimization("alham.png", 1)

if __name__ == "__main__":
	main()
