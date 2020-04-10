# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

from matplotlib import pyplot as plt
import numpy as np
import math
import cv2

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
    # En caso de que los máximos sean 255 o las mínimos 0 no iteramos en los bucles
    if len(image.shape) == 2:
        max = np.amax(image)
        min = np.amin(image)
        if max>255 or min<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    image[i][j] = (image[i][j]-min)/(max-min) * 255
    elif len(image.shape) == 3:
        max = [np.amax(image[:,:,0]), np.amax(image[:,:,1]), np.amax(image[:,:,2])]
        min = [np.amin(image[:,:,0]), np.amin(image[:,:,1]), np.amin(image[:,:,2])]

        if max[0]>255 or max[1]>255 or max[2]>255 or min[0]<0 or min[1]<0 or min[2]<0:
            print("Normalizando imagen '" + image_title + "'")
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        image[i][j][k] = (image[i][j][k]-min[k])/(max[k]-min[k]) * 255

    return image

""" Imprime una imagen a través de una matriz.
- image: imagen a imprimir.
- flag_color (op): bandera para indicar si la imagen es en B/N o color. Por defecto color.
- image_title(op): título de la imagen. Por defecto 'Imagen'
- window_title (op): título de la ventana. Por defecto 'Ejercicio'
"""
def pintaI(image, flag_color=1, image_title = "Imagen", window_title = "Ejercicio"):
    image = normaliza(image, image_title)               # Normalizamos la matriz
    image = image.astype(np.uint8)
    plt.figure(0).canvas.set_window_title(window_title) # Ponemos nombre a la ventana
    if flag_color == 0:
        plt.imshow(image, cmap = "gray")
    else:
        plt.imshow(image)
    plt.title(image_title)              # Ponemos nombre a la imagen
    plt.xticks([])                      # Se le pasa una lista de posiciones en las que se deben colocar los
    plt.yticks([])                      # ticks, si pasamos una lista vacía deshabilitamos los xticks e yticks
    plt.show()
    image = image.astype(np.float64)    # Devolvemos su formato

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
    fig = plt.figure(figsize=(12, 6))
    fig.canvas.set_window_title(window_title)

    for i in range(len(img_list)):
        normaliza(img_list[i], title_list[i])
        img_list[i] = img_list[i].astype(np.uint8)

    for i, a in enumerate(img_list):
        display = fig.add_subplot(1, 3, i + 1) #(1, 4, i + 1)
        if flag_color == 0:
            display.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        else:
            display.imshow(a, interpolation="nearest")
        display.set_title(title_list[i], fontsize=10)
        display.set_xticks([])          # Deshabilitamos xticks
        display.set_yticks([])          # e y_ticks

    fig.tight_layout()
    plt.show()

    for i in range(len(img_list)):
        img_list[i] = img_list[i].astype(np.float64)    # lo devolvemos a su formato.

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
"""
def haar_image(img, img_title):
	print("Calculando la transformada de Haar a la imagen '" + img_title +"'.")
	by_row = haar_row(img)			# por filas
	transpose = zip(*by_row)		# transponemos
	haar_img = haar_row(transpose)	# por columnas
	return haar_img

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
"""
def reverse_image(haar_img, img_title):
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

	return rev_haar

#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    #img_title = input("Image name: ")
    img_title = "BigBoiLion.jpg"
    #img_title = "lena512.bmp"

    img = leer_imagen("images/" + img_title, 0)
    print("La imagen '{}' tiene tamaño {}x{}.".format(img_title, len(img), len(img[0])))

    haar_img = np.array(haar_image(img, img_title))
    rev_img = np.array(reverse_image(haar_img, img_title))

    if (True):
        print("\nMatriz de la imagen original:")
        print(img)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_img)
        print("\nMatriz de la imagen restaurada:")
        print(rev_img)

    show_images([img, haar_img, rev_img],
                ["Original", "2D Haar Transform", "Return to Original"])

if __name__ == "__main__":
	main()
