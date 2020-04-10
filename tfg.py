# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

import numpy as np
from matplotlib import pyplot as plt
import math

import cv2
from PIL import Image

###########################################
###   LECTURA E IMPRESIÓN DE IMÁGENES   ###
###########################################

""" Lee una imagen ya sea en grises o en color. Devuelve la imagen.
- file_name: archivo de la imagen.
- flag_color (op): modo en el que se va a leer la imagen -> grises o color. Por defecto será en color.
"""
def leer_imagen(file_name, flag_color = 1):
    if flag_color == 0:
        print("Leyendo '" + file_name + "' en gris")
    elif flag_color==1:
        print("Leyendo '" + file_name + "' en color")
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

#######################
###   FUNCIONES   ###
#######################

def display_image(img):
# Display a single image
    plt.gray()
    plt.imshow(img)
    plt.show()

def show_images(images):
# Display all the images
    titles = ["Original", "Rows Only", "2D Haar Transform", "Return to Original"]
    fig = plt.figure(figsize=(9, 9)) #(12, 3)
    for i, a in enumerate(images):
        display = fig.add_subplot(2, 2, i + 1) #(1, 4, i + 1)
        display.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        display.set_title(titles[i], fontsize=10)
        display.set_xticks([])
        display.set_yticks([])

    fig.tight_layout()
    plt.show()

""" Realiza el algoritmo Haar Wavelet de manera recursiva realizando las particiones
adecuadas y usando la función 'haar_transform'. Devuelve una lista con la compresión.
- list:
- length:
- offset:
"""
def list_split(list, length, offset):
	if( length > 2 ):
		first = list[:len(list)//2]
		second = list[len(list)//2:]

		first = haar_transform(first)
		offset = second + offset
		return list_split(first, length/2, offset)

	else:
		return list + offset

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

""" Calcula la transformada inversa de Haar. Devuelve la lista resultante.
- front:
- the_rest:
- power:
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
- haar_image: imagen después de la transformada de Haar.
"""
def reverse_img(haar_image):
	rev_columns = []
	power = 0

	# Inversa por columnas
	for pixels in haar_image:
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

#image_chosen = input("Image name: ")
image_chosen = "BigBoiLion.jpg"
image_chosen = "images/" + image_chosen
#resolution = int(input("Dimension of image (256, 512, 1024, 2048, etc.): "))
resolution = 2048
"""
carlos = leer_imagen(image_chosen, 0)
print(len(carlos))
print(len(carlos[0]))
pintaI(carlos, 0)
"""

images_list = []

im = Image.open(image_chosen, 'r')			# opens an image
pix_val = list(im.getdata())				# gets the pixels, stores them in pix_val
line = ','.join(str(v) for v in pix_val)	# turns the list into a string object
string = line.split(',');
nums = [int(i) for i in string]				# nums holds the a list of ints

# Prints original image
og = np.array(pix_val)
img = og.reshape(resolution, resolution)
print(img)
# display_image(img)
images_list.append(img)

# this gives me a list of length of dimension given, each index position holds a list of (256, 512, 1024 ...)
split = list(zip(*[iter(nums)] * resolution))
print("\nNumber of entries in split: " , len(split))
print("length of items in list: " + str(len(split[0])) + "\n")

offset = []
combined = []
print("----> First transform on Rows Only\n:")

for list in split:
    haar = haar_transform(list)
    combined.append(list_split(haar, len(haar), offset))

rows = np.array(combined)
img_rows = rows.reshape(resolution, resolution)
print(img_rows)
# display_image(img_rows)
images_list.append(img_rows)

print("\nThe columns of the first transform are done next")

columns_rows = []
# Transpose of img_rows
transpose_columns = [*zip(*img_rows)]
for column in transpose_columns:
    haar2 = haar_transform(column)
    columns_rows.append(list_split(haar2, len(haar2), offset))

columns = np.array(columns_rows)
haar_image = columns.reshape(resolution, resolution)

print(haar_image)
# display_image(haar_image)
images_list.append(haar_image)

################### transpose the array #########################
rev_img = np.array(reverse_img(haar_image))
#rev_img = rev_image.reshape(resolution, resolution)
print(rev_img)
# display_image(rev_img)
images_list.append(rev_img)
show_images(images_list)
