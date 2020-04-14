# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

from matplotlib import pyplot as plt
import numpy as np
import math
import cv2
import os

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
- image_title (op): título de la imagen. Por defecto ' '.
"""
def normaliza(image, image_title = " "):
    norm = np.copy(image)
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max>255 or min<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    norm[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = [np.amax(image[:,:,0]), np.amax(image[:,:,1]), np.amax(image[:,:,2])]
        min = [np.amin(image[:,:,0]), np.amin(image[:,:,1]), np.amin(image[:,:,2])]

        if max[0]>255 or max[1]>255 or max[2]>255 or min[0]<0 or min[1]<0 or min[2]<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        norm[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

    return norm

""" Imprime una imagen a través de una matriz.
- image: imagen a imprimir.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title(op): título de la imagen. Por defecto 'Imagen'
- window_title (op): título de la ventana. Por defecto 'Ejercicio'
"""
def show_img(image, flag_color=1, image_title = "Imagen", window_title = "Ejercicio"):
    img_show = normaliza(image, image_title)            # Normalizamos la matriz
    img_show = img_show.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(window_title) # Ponemos nombre a la ventana
    if flag_color == 0:
        plt.imshow(img_show, cmap = "gray")
    else:
        plt.imshow(img_show)
    plt.title(image_title)              # Ponemos nombre a la imagen
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

#######################
###   FUNCIONES   ###
#######################

""" Calcula la transformada de Haar de una lista. Devuelve la transformada.
- list: lista sobre la que realizar el cálculo.
"""
def haar_transform(list):
	avgs = []
	difs = []

	for i in range(len(list)):
		if (i%2 == 0):
			avgs.append((list[i] + list[i+1]) / 2)
			difs.append((list[i] - list[i+1]) / 2)

	return avgs + difs

""" Realiza el algoritmo Haar Wavelet de manera recursiva realizando las particiones
adecuadas y usando la función 'haar_transform'. Devuelve una lista con la compresión.
- list: lista a la que hacer el split de haar.
- offset: parte fija del split.
"""
def haar_split(list, offset):
	if(len(list) > 2):
		first = list[:len(list)//2]
		second = list[len(list)//2:]

		first = haar_transform(first)
		offset = second + offset
		return haar_split(first, offset)

	else:
		return list + offset

""" Calcula la transformada de Haar por filas de una imágen.
- img: imagen original a transformar.
"""
def haar_row(img):
	haar_row = []
	for pixels in img:
		haar = haar_transform(pixels)
		haar_row.append(haar_split(haar, []))
	return haar_row

""" Calcula la transformada de Haar una imágen.
- img: imagen original a transformar.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def haar_image(img, img_title="Imagen"):
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
		reverse.append(front[i] + the_rest[i])
		reverse.append(front[i] - the_rest[i])

	if(len(the_rest) > len(reverse) ):
		the_rest = the_rest[2**power:]
		power += 1
		return reverse_haar(reverse, the_rest, power)
	else:
		return reverse

""" Dada una transformada de Haar de una imagen calcula su inversa.
Devuelve la imagen original.
- haar_img: imagen después de la transformada de Haar.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def reverse_image(haar_img, img_title="Imagen"):
    if(img_title != ""):
        print("Restaurando la imagen original de la transformada de Haar de '" + img_title + "'.")

    if(len(haar_img.shape)==2):
        rev_columns = []
        power = 0

        # Inversa por columnas
        for pixels in haar_img:
            rev_columns.append(reverse_haar(pixels[:1], pixels[1:], power))

        rev_columns = zip(*rev_columns)
        rev_haar = []

        # Inversa por filas
        for pixels in rev_columns:
            rev_haar.append(reverse_haar(pixels[:1], pixels[1:], power))

        rev_haar = np.array(rev_haar)
        rev_haar = np.transpose(rev_haar)

        #rev_haar = normaliza(np.array(rev_haar))
        #rev_haar = rev_haar.astype(np.uint8)
        #rev_haar = rev_haar.astype(np.float64)

    else:
        rev_haar = np.zeros(haar_img.shape)
        for k in range(3):
            rev_haar[:,:,k] = reverse_image(haar_img[:,:,k], "")

    return rev_haar

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el threesholding.
- haar_img: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def threesholding(haar_img, epsilon, img_title="Imagen"):
    if(img_title != ""):
        print("Aplicando algoritmo threesholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, img_title))
    threeshold_img = haar_img.copy()
    count = 0

    if(len(haar_img.shape)==2):
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if (abs(haar_img[i][j]) * norm_pixel(i,j) < epsilon):
                    threeshold_img[i][j] = 0.0
                    count += 1

        print("Número de píxeles anulados: {} ({}%)."
            .format(count, round(100*count/(len(haar_img)*len(haar_img[0])), 2)))

    else:
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                for k in range(len(haar_img[0][0])):
                    if (abs(haar_img[i][j][k]) * norm_pixel(i,j) < epsilon):
                        threeshold_img[i][j][k] = 0.0
                        count += 1

        print("Número de píxeles anulados: {} ({}%)."
            .format(count, round(100*count/(len(haar_img)*len(haar_img[0])*len(haar_img[0][0])), 2)))

    return threeshold_img

""" Se queda con la mejor aproximación de m-términos.
Devuelve la imagen después de aplicar el algoritmo.
- haar_img: imagen después de la transformada de Haar.
- m: número de términos que nos vamos a quedar.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def m_term(haar_img, m, img_title="Imagen"):
    if(img_title != ""):
        print("Aplicando algoritmo de m-términos (m={}) a la transformada de Haar de '{}'."
              .format(m, img_title))

    if(len(haar_img.shape)==2):
        list = np.zeros(m)

        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if(m>0):
                    list[i+j] = abs(haar_img[i][j]) * norm_pixel(i,j)
                    m -= 1
                else:
                    val = abs(haar_img[i][j]) * norm_pixel(i,j)
                    if(np.amin(list) < val):
                        list[list==np.amin(list)] = val

        m_term_img = threesholding(haar_img, np.amin(list), "")


    else:
        m_term_img = np.zeros(haar_img.shape)
        for k in range(3):
            m_term_img[:,:,k] = m_term(haar_img[:,:,k], m, "")

    return m_term_img

def m_term_malo(haar_img, m, img_title="Imagen"):
    # SIN ACABAR
    print("Aplicando algoritmo de m-términos a la transformada de Haar de '" + img_title + "'.")
    m_term_img = haar_img.copy()
    delete = len(haar_img)*len(haar_img[0]) - m

    while (delete>0):
        min = 100; x_min=0; y_min=0

        for i in range(len(m_term_img)):
            for j in range(len(m_term_img[0])):
                if abs(m_term_img[i][j]) < min:
                    min = m_term_img[i][j]
                    x_min = i
                    y_min = j
        m_term_img[x_min][y_min] = 0.0
        if (min==100):
            delete = 0
        else:
            delete -= 1

    print("Número de píxeles anulados: {} ({}%)."
        .format(delete, round(100*delete/(len(haar_img)*len(haar_img[0])), 2)))

    return m_term_img

""" Calcula el error medio de la imagen original y su aproximación.
Devuelve el error medio.
- img: imagen original.
- back_img: imagen aproximada.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def error(img, back_img, img_title="Imagen"):
    print("Calculando el error medio de '" + img_title + "' y su aproximación.")
    error = 0

    if(len(img.shape)==2):
        for i in range(len(img)):
            for j in range(len(img[0])):
                error += abs(img[i][j]-back_img[i][j])
        error = error / (len(img)*len(img[0]))
    else:
        for i in range(len(img)):
            for j in range(len(img[0])):
                for k in range(len(img[0][0])):
                    error += abs(img[i][j][k]-back_img[i][j][k])
        error = error / (len(img)*len(img[0])*len(img[0][0]))

    return error

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- img: imagen a recortar.
- sq (op): indica si extender de manera cuadrada. Por defecto False.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_img(img, sq=False, img_title="Imagen"):
    p = 2**int(math.log(len(img), 2))
    q = 2**int(math.log(len(img[0]), 2))

    if(sq):     # ajustamos p y q en caso de ser cuadrada
        if(p<q): q=p
        else: p=q

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
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def extend_img(img, sq=False, img_title="Imagen"):
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
            print("Extendiendo imagen '{}' a tamaño ({}, {}, 3).".format(img_title, p, q))
            ext = np.zeros((p, q, 3), dtype=np.float64)
        else:
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
- rows:
- cols:
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_size(img, rows, cols, img_title="Imagen"):
    print("Recortando imagen '{}' a tamaño {}.".format(img_title, img.shape))
    return img[:rows, :cols]

""" Imprime por pantalla el tamaño en bytes de los archivos.
- file_org: archivo original.
- file_rev: arhivo greedy.
"""
def diff_size(file_org, file_rev):
    print("El archivo original pesa {} bytes y el greedy {} bytes."
          .format(os.stat(file_org).st_size, os.stat(file_rev).st_size))

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

#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    #img_title = "lena512.bmp"
    #img_title = "BigBoiLion.jpg"
    img_title = "alham_sq.png"
    #img_title = "alham.jpg"
    #img_title = "alham_crop.jpg"
    #img_title = "al.jpg"
    epsilon = 0.1
    m = 100000

    img = read_img("images/" + img_title, 1)
    print("Tamaño de la imagen: {}.".format(img.shape))
    ext = extend_img(img, False, img_title)

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, img_title)
    # Aplicándole el algoritmo greedy
    #greedy_img = threesholding(haar_img, epsilon, img_title)
    greedy_img = m_term(haar_img, m, img_title)
    # Restaurando la imagen original
    rev_img = reverse_image(greedy_img, img_title)

    if (True):
        print("\nMatriz de la imagen original:")
        print(img)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_img)
        print("\nMatriz después del algoritmo greedy:")
        print(greedy_img)
        print("\nMatriz de la imagen restaurada:")
        print(rev_img)
        print()

    # Calulamos el error medio de la imagen original y la greedy
    err = error(img, rev_img, img_title)
    print("Error medio: {}.".format(round(err, 4)))

    if(rev_img.shape != img.shape):
        rev_img = crop_size(rev_img, img.shape[0], img.shape[1], img_title)

    # Mostramos diferentes imágenes
    #show_img_list([img, haar_img, greedy_img, rev_img],
    #            ["Original", "2D Haar Transform", "Greedy", "Return to Original"])

    concat_img = concat(img, normaliza(rev_img, img_title))
    show_img(concat_img, 0, img_title, "Haar wavelets")

    save_img("images/rev_" + img_title, rev_img, True)
    save_img("images/greedy_" + img_title, greedy_img, False)
    save_img("images/concat_" + img_title, concat_img, False)

    diff_size("images/" + img_title, "images/greedy_" + img_title)

if __name__ == "__main__":
	main()
