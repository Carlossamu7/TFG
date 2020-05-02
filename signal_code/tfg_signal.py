# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

from matplotlib import pyplot as plt
import numpy as np
import math
from tabulate import tabulate


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

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el thresholding.
- haar_img: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- image_title(op): título de la imagen. Por defecto 'Imagen'
"""
def thresholding(haar_img, epsilon, img_title="Imagen"):
    if(img_title != ""):
        print("Aplicando thresholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, img_title))
    threshold_img = haar_img.copy()
    count = 0
    not_zero = 0

    if(len(haar_img.shape)==2):
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                if(haar_img[i][j]!=0.0):
                    not_zero += 1
                if (abs(haar_img[i][j]) < epsilon):
                    threshold_img[i][j] = 0.0
                    count += 1

        total = len(haar_img)*len(haar_img[0])

    else:
        for i in range(len(haar_img)):
            for j in range(len(haar_img[0])):
                for k in range(len(haar_img[0][0])):
                    if(haar_img[i][j][k]!=0.0):
                        not_zero += 1
                    if (abs(haar_img[i][j][k]) < epsilon):
                        threshold_img[i][j][k] = 0.0
                        count += 1

        total = len(haar_img)*len(haar_img[0])*len(haar_img[0][0])

    perc = round(100*count/total, 2)
    ratio = round(not_zero/(total-count), 4)

    if(img_title != ""):
        print("Número de píxeles anulados: {} ({}%).".format(count, perc))
        print("Ratio de dispersión: {}.".format(ratio))

    return threshold_img, perc, ratio

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

        m_term_img, perc, ratio = thresholding(haar_img, np.amin(list), "")


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
def experiment(img_title, flag, fun, param, print_mat = False, show_im = True, save_im = True):
    print("\n###############################################")
    print("\tTranformada de Haar de {}".format(img_title))
    print("###############################################\n  ")
    img = read_img("images/" + img_title, flag)
    print("Tamaño de la imagen: {}.".format(img.shape))
    ext = extend_img(img, False, img_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_img = haar_image(ext, img_title)
    # Aplicándole el algoritmo greedy
    greedy_img, perc, ratio = fun(haar_img, param, img_title)
    # Comprimimos
    comp_img, cent = compress_img(greedy_img, img_title)

    """
    AQUÍ REALIZARÍAMOS EL ENVÍO DE 'comp'
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

    factor = diff_size(img, comp_img)

    return comp_img, perc, ratio, err, factor

###########################
###   COMPRESSION RLE   ###
###########################

""" Comprime la imagen greedy_img.
Devuelve la compresión como lista y el valor centinela.
- greedy_img: imagen de los coeficientes greedy.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def compress_img(greedy_img, img_title="Imagen"):
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
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def uncompress_img(lists, cent, img_title="Imagen"):
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
    greedy_img, perc, ratio = thresholding(haar_img, thr, "")
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
    print("\n###############################################")
    print("\tOptimizando umbral de {}".format(img_title))
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

    return opt_thr, kneedle.knee


#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    N = 3
    list = [['Ejemplo', 'Umbral', 'Descartes (%)', 'Ratio dispersión', 'Error medio', 'Factor de compresión'],
         ['Lena', 3.0],
         ['León', 3.0],]

    per = np.zeros(N); rat = np.zeros(N); err = np.zeros(N); fac = np.zeros(N)

    _, per[0], rat[0], err[0], fac[0] = experiment("lena.png", 0, thresholding, 3.0, show_im=False, save_im=False)
    _, per[1], rat[1], err[1], fac[1] = experiment("lion.png", 0, thresholding, 3.0, show_im=False, save_im=False)

    for k in range(1,N+1):
        list[k].append(per[k-1])
        list[k].append(rat[k-1])
        list[k].append(err[k-1])
        list[k].append(fac[k-1])

    print()
    print(tabulate(list, headers='firstrow', tablefmt='fancy_grid'))

    #optimization("lena.png", 0)
    #optimization("alham.png", 1)

if __name__ == "__main__":
	main()
