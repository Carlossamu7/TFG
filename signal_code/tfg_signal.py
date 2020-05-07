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

""" Cuenta el número de elementos distintos de cero de una imagen
Devuelve la imagen después de aplicar el thresholding.
- img: imagen sobre la que calcular el número de elementos distintos de cero.
- rows: filas para el conteo.
- cols: columnas para el conteo.
"""
def not_zero(img, rows, cols):
    cont = 0

    for i in range(rows):
            if(img[i]!=0):
                cont += 1

    return cont

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la imagen después de aplicar el thresholding.
- haar_signal: imagen después de la transformada de Haar.
- epsilon: valor umbral.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def thresholding(haar_signal, epsilon, signal_title="Imagen"):
    if(signal_title != ""):
        print("Aplicando thresholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, signal_title))
    threshold_signal = haar_signal.copy()

    for i in range(len(haar_signal)):
        if (abs(haar_signal[i]) < epsilon):
            threshold_signal[i] = 0.0

    return threshold_signal

""" Se queda con la mejor aproximación de m-términos.
Devuelve la imagen después de aplicar el algoritmo.
- haar_signal: imagen después de la transformada de Haar.
- m: número de términos que nos vamos a quedar.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def m_term(haar_signal, m, signal_title="Imagen"):
    if(signal_title != ""):
        print("Aplicando algoritmo de m-términos (m={}) a la transformada de Haar de '{}'."
              .format(m, signal_title))

    list = np.zeros(m)

    for i in range(len(haar_signal)):
        for j in range(len(haar_signal[0])):
            if(m>0):
                list[i] = abs(haar_signal[i])
                m -= 1
            else:
                val = abs(haar_signal[i])
                if(np.amin(list) < val):
                    list[list==np.amin(list)] = val

    m_term_signal = thresholding(haar_signal, np.amin(list), "")

    return m_term_signal

""" Calcula el error medio de la imagen original y su aproximación.
Devuelve el error medio.
- signal: imagen original.
- back_signal: imagen aproximada.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def error(signal, back_signal, signal_title="Imagen"):
    err = 0

    for i in range(len(signal)):
        for j in range(len(signal[0])):
            err += abs(signal[i]-back_signal[i])
    err = err / (len(signal)*len(signal[0]))

    err = round(err, 4)
    if(signal_title != ""):
        print("Error medio de '{}' y su aproximación: {}.".format(signal_title, err))

    return err

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- signal: imagen a recortar.
- signal_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_signal(signal, sq=False, signal_title="Imagen"):
    p = 2**int(math.log(len(signal), 2))

    if(signal_title != ""):
        print("Recortando imagen '{}' a tamaño {}.".format(signal_title, p))

    a = int((len(signal)-p)/2)

    return signal[a:(a+p)]

""" Extiende a la imagen de tamaño 2^p*2^q más pequeña dentro de la que cabe.
Devuelve la imagen extendida.
- signal: imagen a extender.
- sq (op): indica si extender de manera cuadrada. Por defecto False.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def extend_signal(signal, sq=False, signal_title="Imagen"):
    n = math.log(len(signal), 2)
    to_extend = False

    if(int(n)<n):
        n = int(n) + 1
        to_extend = True
    else:
        n = int(n)

    if(to_extend):
        p = 2**n

        if(signal_title != ""):
            print("Extendiendo señal '{}' a tamaño {}.".format(signal_title, p))
        ext = np.zeros((p), dtype=np.float64)

        for i in range(len(signal)):
            ext[i] = signal[i]

        return ext
    else:
        return signal

""" Recorta una imagen a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la imagen recortada.
- signal: imagen a recortar.
- size: tamaño al que recortar.
- signal_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def crop_size(signal, crop_size, signal_title="Imagen"):
    if(signal_title != ""):
        print("Recortando imagen '{}' a tamaño {}.".format(signal_title, size))
    return signal[:size]

""" Imprime por pantalla el tamaño en bytes de los archivos y el factor de compresión.
Devuelve el factor de compresión.
- file_org: archivo original.
- file_rev: arhivo greedy.
"""
def diff_size(signal, comp_signal):
    comp_factor = signal.nbytes / np.array(comp_signal).nbytes
    print("El archivo original pesa {} bytes y la compresión del greedy {} bytes."
          .format(signal.nbytes, np.array(comp_signal).nbytes))
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

""" Experimento greedy a realizar.
Devuelve la compresión, porcentaje de descarte, ratio de dispersión,
error medio y factor de compresión.
- signal_title: título de la imagen.
- flag: 0 para B/N y 1 color.
- fun: función de aproximación (thresholding, m_term).
- param: parámetro de la función de aproximación.
- print_mat (op): indica si se deben imprimir las matrices. Por defecto 'False'.
- show_im (op): indica si mostrar las imágenes. Por defeto 'True'.
- save_im (op): indica si guardar las imágenes. Por defecto 'True'.
"""
def experiment(signal_title, flag, fun, param, print_mat = False, show_im = True, save_im = True):
    print("\n###############################################")
    print("\tTranformada de Haar de {}".format(signal_title))
    print("###############################################\n  ")
    # SEÑAL
    size = 1024
    print("Tamaño de la señal: {}.".format(size))
    ext = extend_signal(signal, False, signal_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_signal = haar_image(ext, signal_title)
    # Aplicándole el algoritmo greedy
    not_zero_before = not_zero(haar_signal, len(signal))
    greedy_signal = fun(haar_signal, param, signal_title)
    not_zero_after = not_zero(greedy_signal, len(signal))
    # Calulando ratio y perc
    if(len(signal.shape)==2):
        total = len(signal)*len(signal[0])
    else:
        total = len(signal)*len(signal[0])*len(signal[0][0])
    perc = round(100*(total-not_zero_after)/total, 2)
    ratio = round(not_zero_before/not_zero_after, 4)
    if(signal_title != ""):
        print("Número de píxeles anulados: {} ({}%).".format(total-not_zero_after, perc))
        print("Ratio de dispersión: {}.".format(ratio))
    # Comprimimos
    comp_signal, cent = compress_signal(greedy_signal, signal_title)

    """
    AQUÍ REALIZARÍAMOS EL ENVÍO DE 'comp_signal'
    """

    # Descomprimimos
    uncomp_signal = uncompress_signal(comp_signal, cent, signal_title)
    # Restaurando la imagen original
    rev_signal = reverse_image(uncomp_signal, signal_title)

    if(rev_signal.shape != signal.shape): # recorta si hemos extendido
        rev_signal = crop_size(rev_signal, signal.shape[0], signal.shape[1], signal_title)

    if (print_mat):  # si queremos imprimir las matrices
        print("\nMatriz de la imagen original:")
        print(signal)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_signal)
        print("\nMatriz después del algoritmo greedy:")
        print(greedy_signal)
        print("\nMatriz de la descompresión:")
        print(uncomp_signal)
        print("\nMatriz de la imagen restaurada:")
        print(rev_signal)
        print()

    # Calulamos el error medio de la imagen original y la revertida
    err = error(signal, rev_signal, signal_title)

    # Concatenamos la imagen original y la recuperada para pintarla
    concat_signal = concat(signal, normaliza(rev_signal, signal_title))
    if(show_im):
        show_signal(concat_signal, flag, signal_title, "Haar wavelets")

    if(save_im):    # Guardamos las imágenes
        save_signal("results/rev" + str(param) + "_" + signal_title, rev_signal, True)
        save_signal("results/greedy" + str(param) + "_" + signal_title, greedy_signal, False)
        save_signal("results/concat" + str(param) + "_" + signal_title, concat_signal, False)

    factor = diff_size(signal, comp_signal)

    return comp_signal, perc, ratio, err, factor

###########################
###   COMPRESSION RLE   ###
###########################

""" Comprime la imagen greedy_signal.
Devuelve la compresión como lista y el valor centinela.
- greedy_signal: imagen de los coeficientes greedy.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def compress_signal(greedy_signal, signal_title="Imagen"):
    comp = []

    if(signal_title != ""):
        print("Comprimiendo imagen de coeficientes de '{}'.".format(signal_title))

    if(len(greedy_signal.shape)==2):
        cent = int(np.amax(greedy_signal)) + 1

        for i in range(len(greedy_signal)):
            row = []
            count = 0

            for j in range(len(greedy_signal[0])-1):
                if(greedy_signal[i][j]==0):
                    count +=1
                elif(count>0):
                    row.append(cent)
                    row.append(count)
                    count = 0
                    row.append(greedy_signal[i][j])
                else:
                    row.append(greedy_signal[i][j])

            if(count>0):
                if(greedy_signal[i][-1]==0):
                    row.append(cent)
                    row.append(count+1)
                else:
                    row.append(cent)
                    row.append(count)
                    row.append(greedy_signal[i][-1])
            else:
                row.append(greedy_signal[i][-1])

            comp.append(row)

    else:
        cent = []

        for k in range(len(greedy_signal[0][0])):
            co, ce = compress_signal(greedy_signal[:,:,k], "")
            comp.append(co)
            cent.append(ce)

    return comp, cent

""" Descomprime una imagen comprimida. Devuelve la imagen.
- lists: compresión.
- cent: valor centinela.
- image_title(op): título de la imagen. Por defecto 'Imagen'.
"""
def uncompress_signal(lists, cent, signal_title="Imagen"):
    if(signal_title != ""):
        print("Descomprimiendo imagen '{}'.".format(signal_title))

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
        signal = np.array(un)

    else:   # Color
        for k in range(len(cent)):
            un = uncompress_signal(lists[k], cent[k], "")
            if(k==0):
                signal = np.empty((un.shape[0], un.shape[1], 3))
            signal[:,:,k] = un

    return signal

########################
###   OPTIMIZATION   ###
########################

""" Experimento greedy a realizar.
Devuelve el error, porcentaje de descarte y ratio de descompresión obtenidos.
- signal: imagen inicial sobre la que realizar el experimento.
- thr: parámetro de la función de aproximación.
"""
def experiment_opt(signal, thr):
    ext = extend_signal(signal, False, "") # solo la extiende si es necesario

    # Calculando la transformada de Haar
    haar_signal = haar_image(ext, "")
    # Aplicándole el algoritmo greedy
    greedy_signal, perc, ratio = thresholding(haar_signal, thr, "")
    # Restaurando la imagen original
    rev_signal = reverse_image(greedy_signal, "")

    if(rev_signal.shape != signal.shape): # recorta si hemos extendido
        rev_signal = crop_size(rev_signal, signal.shape[0], signal.shape[1], "")

    # Calulamos el error medio de la imagen original y la revertida
    err = error(signal, rev_signal, "")
    return err, perc, ratio

""" Optimización del error medio. Devuelve el punto 'knee'.
- signal_title: título de la imagen.
- flag: 0 para B/N y 1 color.
"""
def optimization(signal_title, flag):
    print("\n###############################################")
    print("\tOptimizando umbral de {}".format(signal_title))
    print("###############################################\n  ")
    signal = read_signal("images/" + signal_title, flag)
    print("Tamaño de la imagen: {}.".format(signal.shape))
    thrs = []; errs = []; pers = []; rats = []

    for thr in range(1,20,1):
        err, per, rat = experiment_opt(signal, thr)
        thrs.append(thr); errs.append(err); pers.append(per); rats.append(rat)
    for thr in range(20,40,2):
        err, per, rat = experiment_opt(signal, thr)
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
    plt.title("Relación porcentaje de descarte - error para '{}'".format(signal_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("results/graf_pers_" + signal_title)
    plt.show()

    plt.plot(thrs, errs, '-o', linewidth=1)
    plt.vlines(opt_thr, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Umbral del punto 'knee'")
    plt.xlabel("Umbral")
    plt.ylabel("Error medio")
    plt.legend(loc="lower right")
    plt.title("Relación umbral - error para '{}'".format(signal_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("results/graf_thrs_" + signal_title)
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
