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
def leer_imagen(file_name, flag_color = 1):
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
def pintaI(image, flag_color=1, image_title = "Imagen", window_title = "Ejercicio"):
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
def leer_lista_imagenes(file_name_list, flag_color = 1):
    image_list = []

    for i in file_name_list:
        img = leer_imagen(i, flag_color)
        image_list.append(img)

    return image_list

""" Muestra múltiples imágenes en una ventena Matplotlib.
- img_list: La lista de imágenes.
- title_list: Lista de títulos de las imágenes.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- window_title (op): título de la ventana. Por defecto 'Imágenes con títulos'
"""
def show_images(img_list, title_list, flag_color=0, window_title="Haar Wavelets"):
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
"""
def save_img(file_name, img):
    print("Guardando imagen '" + file_name + "'.")
    cv2.imwrite(file_name, img)

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
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def haar_image(img, img_title="Imagen"):
	print("Calculando la transformada de Haar a la imagen '" + img_title +"'.")
	by_row = haar_row(img)			# por filas
	transpose = zip(*by_row)		# transponemos
	haar_img = haar_row(transpose)	# por columnas
	return np.array(haar_img)

""" Calcula la transformada inversa de Haar. Devuelve la lista resultante.
- front: primera parte de la lista.
- the_rest: segunda parte de la lista.
- power: 2 elevadoa este exponente me dice el índice de the_rest en la lista.
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
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def reverse_image(haar_img, img_title="Imagen"):
    print("Restaurando la imagen original de la transformada de Haar de '" + img_title + "'.")
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

    #rev_haar = normaliza(np.array(rev_haar))
    #rev_haar = rev_haar.astype(np.uint8)
    #rev_haar = rev_haar.astype(np.float64)

    return np.array(rev_haar)

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el threesholding.
- haar_img: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def threesholding(haar_img, epsilon, img_title="Imagen"):
    print("Aplicando algoritmo threesholding a la transformada de Haar de '" + img_title + "'.")
    threeshold_img = haar_img.copy()
    count = 0

    for i in range(len(haar_img)):
        for j in range(len(haar_img[0])):
            if abs(haar_img[i][j]) < epsilon:
                threeshold_img[i][j] = 0.0
                count += 1

    print("Número de píxeles anulados: {} ({}%)."
        .format(count, round(100*count/(len(haar_img)*len(haar_img[0])), 2)))

    return threeshold_img

""" Se queda con la mejor aproximación de m-términos.
Devuelve la imagen después de aplicar el algoritmo.
- haar_img: imagen después de la transformada de Haar.
- m: número de términos que nos vamos a quedar.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def m_term(haar_img, m, img_title="Imagen"):
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
    for i in range(len(img)):
        for j in range(len(img[0])):
            error += abs(img[i][j]-back_img[i][j])
    return error / (len(img)*len(img[0]))

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- img: imagen a recortar.
- sq (op): indica si extender de manera cuadrada. Por defecto False.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_img(img, sq=False, img_title="Imagen"):
    p = 2**int(math.log(len(img), 2))
    q = 2**int(math.log(len(img[0]), 2))
    print(p)
    print(q)

    if(sq):     # ajustamos p y q en caso de ser cuadrada
        if(p<q): q=p
        else: p=q

    if(len(img.shape)==3):
        print("Recortando imagen '{}' a tamaño {}x{}x3".format(img_title, p, q))
    else:
        print("Recortando imagen '{}' a tamaño {}x{}".format(img_title, p, q))

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

    if(int(n)<n): n = int(n) + 1
    else: n = int(n)
    if(int(m)<m): m = int(m) + 1
    else: m = int(m)

    p = 2**n
    q = 2**m

    if(sq):     # ajustamos p y q en caso de ser cuadrada
        if(p>q): q=p
        else: p=q

    if(len(img.shape)==3):
        print("Extendiendo imagen '{}' a tamaño {}x{}x3".format(img_title, p, q))
        ext = np.zeros((p, q, 3), dtype=np.uint8)
    else:
        print("Extendiendo imagen '{}' a tamaño {}x{}".format(img_title, p, q))
        ext = np.zeros((p, q), dtype=np.uint8)

    for i in range(len(img)):
        for j in range(len(img[0])):
            ext[i][j] = img[i][j]

    return ext

""" Imprime por pantalla el tamaño en bytes de los archivos.
- file_org: archivo original.
- file_rev: arhivo greedy.
"""
def diff_size(file_org, file_rev):
    print("El archivo original pesa {} bytes y el greedy {} bytes."
          .format(os.stat(file_org).st_size, os.stat(file_rev).st_size))


#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    #img_title = input("Image name: ")
    img_title = "BigBoiLion.jpg"
    #img_title = "lena512.bmp"
    #img_title = "alham.jpg"
    epsilon = 1.0
    m = 1000000

    img = leer_imagen("images/" + img_title, 0)

    # Calculando la transformada de Haar
    haar_img = haar_image(img, img_title)
    # Aplicándole el algoritmo greedy
    greedy_img = threesholding(haar_img, epsilon, img_title)
    #greedy_img = np.array(m_term(haar_img, m, img_title))
    # Restaurando la imagen original
    rev_img = reverse_image(greedy_img, img_title)

    #greedy_img = greedy_img.astype(np.uint8)
    #greedy_img = greedy_img.astype(np.float64)

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

    # Mostramos diferentes imágenes
    show_images([img, haar_img, greedy_img, rev_img],
                ["Original", "2D Haar Transform", "Greedy", "Return to Original"])

    save_img("images/rev_" + img_title, rev_img)
    save_img("images/greedy_" + img_title, greedy_img)

    diff_size("images/" + img_title, "images/greedy_" + img_title)

if __name__ == "__main__":
	main()
