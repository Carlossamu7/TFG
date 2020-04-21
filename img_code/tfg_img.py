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
from kneed import KneeLocator

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
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def reverse_row(haar_img, img_title="Imagen"):
    row = []
    for pixels in haar_img:
        row.append(reverse_haar(pixels[:1], pixels[1:], 0))
    return row

""" Dada una transformada de Haar de una imagen calcula su inversa.
Devuelve la imagen original.
- haar_img: imagen después de la transformada de Haar.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def reverse_image(haar_img, img_title="Imagen"):
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

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el threesholding.
- haar_img: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def threesholding(haar_img, epsilon, img_title="Imagen"):
    if(img_title != ""):
        print("Aplicando threesholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, img_title))
    threeshold_img = haar_img.copy()
    count = 0
    not_zero = 0

    if(len(haar_img.shape)==2):
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if(haar_img[i][j]!=0.0):
                    not_zero += 1
                if (abs(haar_img[i][j]) < epsilon):
                    threeshold_img[i][j] = 0.0
                    count += 1

        total = len(haar_img)*len(haar_img[0])

    else:
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                for k in range(len(haar_img[0][0])):
                    if(haar_img[i][j][k]!=0.0):
                        not_zero += 1
                    if (abs(haar_img[i][j][k]) < epsilon):
                        threeshold_img[i][j][k] = 0.0
                        count += 1

        total = len(haar_img)*len(haar_img[0])*len(haar_img[0][0])

    perc = round(100*count/total, 2)
    ratio = round(not_zero/(total-count), 4)

    if(img_title != ""):
        print("Número de píxeles anulados: {} ({}%).".format(count, perc))
        print("Ratio de dispersión: {}.".format(ratio))

    return threeshold_img, perc, ratio

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
                    list[i+j] = abs(haar_img[i][j])
                    m -= 1
                else:
                    val = abs(haar_img[i][j])
                    if(np.amin(list) < val):
                        list[list==np.amin(list)] = val

        m_term_img, perc, ratio = threesholding(haar_img, np.amin(list), "")


    else:
        m_term_img = np.zeros(haar_img.shape)
        perc = 0
        ratio = 0
        for k in range(3):
            m_term_img[:,:,k], per, rat = m_term(haar_img[:,:,k], m, "")
            perc += per
            ratio += rat

    if(img_title != ""):
        print("Porcentaje de píxeles descartados: {}%.".format(perc))
        print("Ratio de dispersión: {}.".format(ratio))

    return m_term_img, perc, ratio

""" Calcula el error medio de la imagen original y su aproximación.
Devuelve el error medio.
- img: imagen original.
- back_img: imagen aproximada.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def error(img, back_img, img_title="Imagen"):
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
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_img(img, sq=False, img_title="Imagen"):
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
- rows:
- cols:
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_size(img, rows, cols, img_title="Imagen"):
    if(img_title != ""):
        if(len(img.shape)==2):
            print("Recortando imagen '{}' a tamaño ({}, {}).".format(img_title, rows, cols))
        else:
            print("Recortando imagen '{}' a tamaño ({}, {}, {}).".format(img_title, rows, cols, 3))
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
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
- fun: función de aproximación (threesholding, m_term).
- param: parámetro de la función de aproximación.
- print_mat (op): indica si se deben imprimir las matrices. Por defecto 'False'.
- show_im (op): indica si mostrar las imágenes. Por defeto 'True'.
- save_im (op): indica si guardar las imágenes. Por defecto 'True'.
"""
def experiment(img_title, flag, fun, param, print_mat = False, show_im = True, save_im = True):
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    ext = extend_img(img, False, img_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, img_title)
    # Aplicándole el algoritmo greedy
    greedy_img, perc, ratio = fun(haar_img, param, img_title)
    # Restaurando la imagen original
    rev_img = reverse_image(greedy_img, img_title)

    if(rev_img.shape != img.shape): # recorta si hemos extendido
        rev_img = crop_size(rev_img, img.shape[0], img.shape[1], img_title)

    if (print_mat):  # si queremos imprimir las matrices
        print("\nMatriz de la imagen original:")
        print(img)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_img)
        print("\nMatriz después del algoritmo greedy:")
        print(greedy_img)
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
        show_img(concat_img, 0, img_title, "Haar wavelets")

    if(save_im):    # Guardamos las imágenes
        save_img("images/rev" + str(param) + "_" + img_title, rev_img, True)
        save_img("images/greedy" + str(param) + "_" + img_title, greedy_img, False)
        #save_img("images/concat" + str(param) + "_" + img_title, concat_img, False)
        diff_size("images/" + img_title, "images/greedy_" + img_title)

    return err, perc, ratio

""" Obtenemos el gradiente de la imagen
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
"""
def getDerivates(img_title, flag):
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
    save_img("images/der_" + img_title, der)

    return der

""" Experimento greedy a realizar.
- img: imagen inicial sobre la que realizar el experimento.
- thr: parámetro de la función de aproximación.
"""
def experiment_opt(img, thr):
    ext = extend_img(img, False, "") # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, "")
    # Aplicándole el algoritmo greedy
    greedy_img, perc, ratio = threesholding(haar_img, thr, "")
    # Restaurando la imagen original
    rev_img = reverse_image(greedy_img, "")

    if(rev_img.shape != img.shape): # recorta si hemos extendido
        rev_img = crop_size(rev_img, img.shape[0], img.shape[1], "")

    # Calulamos el error medio de la imagen original y la revertida
    err = error(img, rev_img, "")
    return err, perc, ratio

""" Optimización del error medio. Devuelve el punto 'knee'.
- img_title: título de la imagen.
- flag: 0 para B/N y 1 color.
"""
def optimization(img_title, flag):
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    thrs = []; errs = []; pers = []; rats = []

    for thr in range(0,40,2):
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
    print("El punto 'knee' es: {}".format(round(kneedle.knee, 2)))

    # Busco el umbral asociado a ese 'knee'
    for i in range(len(pers)):
        if (pers[i] == kneedle.knee):
            opt_thr = thrs[i]
    print("El umbral asociado es: {}".format(opt_thr))

    # Imprimo las gráficas
    plt.plot(pers, errs, '-o', linewidth=1)
    plt.vlines(kneedle.knee, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Punto 'knee'")
    plt.xlabel("Porcentaje de descartados")
    plt.ylabel("Error medio")
    plt.legend()
    plt.title("Relación porcentaje de descarte - error para '{}'".format(img_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("images/graf_pers_" + img_title)
    plt.show()

    plt.plot(thrs, errs, '-o', linewidth=1)
    plt.vlines(opt_thr, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Umbral del punto 'knee'")
    plt.xlabel("Umbral")
    plt.ylabel("Error medio")
    plt.legend(loc="lower right")
    plt.title("Relación umbral - error para '{}'".format(img_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("images/graf_thrs_" + img_title)
    plt.show()

    return opt_thr, kneedle.knee

#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    experiment("lena.png", 0, threesholding, 40.0)
    #experiment("lion.png", 0, threesholding, 50.0)
    #experiment("lena_color.png", 1, threesholding, 50.0)
    #experiment("alham.png", 1, threesholding, 40.0)

    getDerivates("lena.png", 0)

    #optimization("lena.png", 0)
    #optimization("alham.png", 1)

if __name__ == "__main__":
	main()
