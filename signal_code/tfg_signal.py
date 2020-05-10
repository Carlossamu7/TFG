# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:54:38 2020
@author: Carlos Sánchez Muñoz
"""

from matplotlib import pyplot as plt
import numpy as np
import math
from kneed import KneeLocator
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
        for i in range(len(list)):
            list[i] = list[i] / math.sqrt(len(list)+len(offset))
        for i in range(len(offset)):
            offset[i] = offset[i] / math.sqrt(len(list)+len(offset))
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
        for i in range(len(reverse)):
            reverse[i] = reverse[i] * math.sqrt(len(reverse))
        return reverse

""" Cuenta el número de elementos distintos de cero de una señal
Devuelve la señal después de aplicar el thresholding.
- signal: señal sobre la que calcular el número de elementos distintos de cero.
- size: tamaño para el conteo.
"""
def not_zero(signal, size):
    cont = 0

    for i in range(size):
            if(signal[i]!=0):
                cont += 1

    return cont

""" Asigna 0 a aquellos elementos que estén por debajo de un umbral.
Devuelve la señal después de aplicar el thresholding.
- haar_signal: señal después de la transformada de Haar.
- epsilon: valor umbral.
- signal_title(op): título de la señal. Por defecto "".
"""
def thresholding(haar_signal, epsilon, signal_title=""):
    if(signal_title != ""):
        print("Aplicando thresholding con epsilon={} a la transformada de Haar de '{}'."
              .format(epsilon, signal_title))
    threshold_signal = haar_signal.copy()

    for i in range(len(haar_signal)):
        if (abs(haar_signal[i]) < epsilon):
            threshold_signal[i] = 0.0

    return threshold_signal

""" Se queda con la mejor aproximación de m-términos.
Devuelve la señal después de aplicar el algoritmo.
- haar_signal: señal después de la transformada de Haar.
- m: número de términos que nos vamos a quedar.
- signal_title(op): título de la señal. Por defecto "".
"""
def m_term(haar_signal, m, signal_title=""):
    if(signal_title != ""):
        print("Aplicando algoritmo de m-términos (m={}) a la transformada de Haar de '{}'."
              .format(m, signal_title))

    list = np.zeros(m)

    for i in range(len(haar_signal)):
        if(m>0):
            list[i] = abs(haar_signal[i])
            m -= 1
        else:
            val = abs(haar_signal[i])
            if(np.amin(list) < val):
                list[list==np.amin(list)] = val

    m_term_signal = thresholding(haar_signal, np.amin(list), "")

    return m_term_signal

""" Calcula el error medio de la señal original y su aproximación.
Devuelve el error medio.
- signal: señal original.
- back_signal: señal aproximada.
- signal_title(op): título de la señal. Por defecto "".
"""
def error(signal, back_signal, signal_title=""):
    err = 0

    for i in range(len(signal)):
        err += abs(signal[i]-back_signal[i])
    err = err / len(signal)
    err = round(err, 4)
    if(signal_title != ""):
        print("Error medio de '{}' y su aproximación: {}.".format(signal_title, err))

    return err

""" Recorta una señal a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la señal recortada.
- signal: señal a recortar.
- signal_title(op): título de la señal. Por defecto "".
"""
def crop_signal(signal, sq=False, signal_title=""):
    p = 2**int(math.log(len(signal), 2))

    if(signal_title != ""):
        print("Recortando señal '{}' a tamaño {}.".format(signal_title, p))

    a = int((len(signal)-p)/2)

    return signal[a:(a+p)]

""" Extiende a la señal de tamaño 2^p*2^q más pequeña dentro de la que cabe.
Devuelve la señal extendida.
- signal: señal a extender.
- signal_title(op): título de la señal. Por defecto "".
"""
def extend_signal(signal, signal_title=""):
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

""" Recorta una señal a la de tamaño 2^p*2^q más grande dentro de ella.
Devuelve la señal recortada.
- signal: señal a recortar.
- size: tamaño al que recortar.
- signal_title(op): título de la señal. Por defecto "".
"""
def crop_size(signal, size, signal_title=""):
    if(len(signal)!=size):
        if(signal_title != ""):
            print("Recortando señal '{}' a tamaño {}.".format(signal_title, size))
        return signal[:size]
    else:
        return signal

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
Devuelve la compresión, porcentaje de descarte, error medio y factor de compresión.
- signal_f: señal.
- N: número de trozos en los que discretizar la señal.
- fun: función de aproximación (thresholding, m_term).
- dom: dominio de la señal.
- param: parámetro de la función de aproximación.
- signal_title(op): título de la señal. Por defecto "".
- print_mat (op): indica si se deben imprimir las matrices. Por defecto 'False'.
- show_sig (op): indica si mostrar las imágenes. Por defeto 'True'.
- save_sig (op): indica si guardar las imágenes. Por defecto 'True'.
"""
def experiment(signal_f, dom, N, fun, param, signal_title="", print_mat=False, show_sig=True, save_sig=True):
    print("\n#####################################################")
    print("    Tranformada de Haar de {}".format(signal_title))
    print("#####################################################\n  ")
    puntos = np.linspace(dom[0], dom[1], num=N)
    signal = np.empty((N), dtype=np.float64)
    for i in range(N):
        signal[i] = signal_f(puntos[i])
    print("Tamaño de la señal: {}.".format(N))
    ext = extend_signal(signal, signal_title) # solo la extiende si es necesario

    # Calculando la transformada de Haar
    if(signal_title != ""):
        print("Calculando la transformada de Haar de la señal '" + signal_title +"'.")
    haar_signal = haar_transform(ext, [])
    # Aplicándole el algoritmo greedy
    greedy_signal = fun(haar_signal, param, signal_title)
    not_zero_after = not_zero(greedy_signal, len(signal))
    # Calulando porcentaje de descarte
    perc = round(100*(N-not_zero_after)/N, 2)
    if(signal_title != ""):
        print("Número de píxeles anulados: {} ({}%).".format(N-not_zero_after, perc))
    # Comprimimos
    comp_signal, cent = compress_signal(greedy_signal, signal_title)

    """
    AQUÍ REALIZARÍAMOS EL ENVÍO DE 'comp_signal'
    """

    # Descomprimimos
    uncomp_signal = uncompress_signal(comp_signal, cent, signal_title)
    # Restaurando la señal original
    if(signal_title != ""):
        print("Restaurando la señal original de la transformada de Haar de '" + signal_title + "'.")
    rev_signal = reverse_haar(uncomp_signal[:1], uncomp_signal[1:], 0)
    # Recorta si hemos extendido
    rev_signal = crop_size(rev_signal, N, signal_title)

    if (print_mat):  # si queremos imprimir las matrices
        print("\nMatriz de la señal original:")
        print(signal)
        print("\nMatriz después de la transformada de Haar:")
        print(haar_signal)
        print("\nMatriz después del algoritmo greedy:")
        print(greedy_signal)
        print("\nMatriz de la compresión:")
        print(comp_signal)
        print("\nMatriz de la descompresión:")
        print(uncomp_signal)
        print("\nMatriz de la señal restaurada:")
        print(rev_signal)
        print()

    # Calulamos el error medio de la señal original y la revertida
    err = error(signal, rev_signal, signal_title)

    # Construimos la gráfica
    if(show_sig or save_sig):
        plt.plot(puntos, signal, 'k', label="Señal")
        plt.plot(puntos, rev_signal, 'r', label="Aproximación")
        plt.xlabel("Eje x")
        plt.ylabel("Eje y")
        plt.legend(loc="lower left")
        plt.title(signal_title)
        plt.gcf().canvas.set_window_title('TFG')
        if(save_sig):    # Guardar
            plt.savefig("results/graf_" + str(N) + "_" + str(int(10*param)) + "_" + signal_title + ".png")
        if(show_sig):    # Visualizar
            plt.show()

    factor = diff_size(signal, comp_signal)
    return comp_signal, perc, err, factor

###########################
###   COMPRESSION RLE   ###
###########################

""" Comprime la señal greedy_signal.
Devuelve la compresión como lista y el valor centinela.
- greedy_signal: señal de los coeficientes greedy.
- signal_title(op): título de la señal. Por defecto "".
"""
def compress_signal(greedy_signal, signal_title=""):
    comp = []

    if(signal_title != ""):
        print("Comprimiendo señal de coeficientes de '{}'.".format(signal_title))

    cent = int(np.amax(greedy_signal)) + 1
    count = 0

    for i in range(len(greedy_signal)-1):
        if(greedy_signal[i]==0):
            count +=1
        elif(count>0):
            comp.append(cent)
            comp.append(count)
            count = 0
            comp.append(greedy_signal[i])
        else:
            comp.append(greedy_signal[i])

    if(count>0):
        if(greedy_signal[-1]==0):
            comp.append(cent)
            comp.append(count+1)
        else:
            comp.append(cent)
            comp.append(count)
            comp.append(greedy_signal[-1])
    else:
        comp.append(greedy_signal[-1])

    return comp, cent

""" Descomprime una señal comprimida. Devuelve la señal.
- list: compresión.
- cent: valor centinela.
- signal_title(op): título de la señal. Por defecto "".
"""
def uncompress_signal(list, cent, signal_title=""):
    if(signal_title != ""):
        print("Descomprimiendo señal '{}'.".format(signal_title))

    un = []
    act = False

    for j in range(len(list)):
        if(act == True):
            act = False
            for i in range(list[j]):
                un.append(0)
        elif(list[j]!=cent):
            un.append(list[j])
        else:   # list[j]==cent
            act = True

    signal = np.array(un)
    return signal

####################################
###   OPTIMIZATION DE THRESHOLD  ###
####################################

""" Experimento greedy a realizar.
Devuelve el error y porcentaje de descarte.
- signal: señal inicial sobre la que realizar el experimento.
- N: número de trozos en los que discretizar la señal.
- thr: parámetro de la función de aproximación.
"""
def experiment_opt(signal, N, thr):
    ext = extend_signal(signal, "") # solo la extiende si es necesario
    # Calculando la transformada de Haar
    haar_signal = haar_transform(ext, [])
    # Aplicándole el algoritmo greedy
    greedy_signal = thresholding(haar_signal, thr, "")
    not_zero_after = not_zero(greedy_signal, len(signal))
    perc = round(100*(N-not_zero_after)/N, 2)
    # Restaurando la señal original
    rev_signal = reverse_haar(greedy_signal[:1], greedy_signal[1:], 0)

    if(len(rev_signal) != len(signal)): # recorta si hemos extendido
        rev_signal = crop_size(rev_signal, N, "")

    if(False):
        puntos = np.linspace(0, 2*np.pi, num=N)
        plt.plot(puntos, signal, 'k', label="Señal")
        plt.plot(puntos, rev_signal, 'r', label="Aproximación")
        plt.xlabel("Eje x")
        plt.ylabel("Eje y")
        plt.legend(loc="lower left")
        plt.title("Optimización")
        plt.gcf().canvas.set_window_title('TFG')
        plt.show()

    # Calulamos el error medio de la señal original y la revertida
    err = error(signal, rev_signal, "")
    return err, perc

""" Optimización del error medio. Devuelve el punto 'knee', el umbral y error asociado.
- signal_f: señal.
- dom: dominio de la señal.
- N: número de trozos en los que discretizar la señal.
- signal_title: título de la señal.
- show_n_save(op): Flag que indica si mostrar y guardar las gráficas. Por defecto 'True'.
"""
def optimization_thr(signal_f, dom, N, signal_title, show_n_save = True):
    if(signal_title!=""):
        print("\n#####################################################")
        print("    Optimizando umbral de '{}'".format(signal_title))
        print("#####################################################\n  ")
        print("Tamaño de la señal: {}.".format(N))
    puntos = np.linspace(dom[0], dom[1], num=N)
    signal = np.empty((N), dtype=np.float64)
    for i in range(N):
        signal[i] = signal_f(puntos[i])

    thrs = []; errs = []; pers = [];

    for thr in range(1,200,5):
        thr = thr/100
        err, per = experiment_opt(signal, N, thr)
        thrs.append(thr); errs.append(err); pers.append(per);

    if(signal_title!=""): # Imprimo las listas
        print("Umbrales:")
        print(thrs)
        print("Errores:")
        print(errs)
        print("Porcentajes de descarte:")
        print(pers)

    # Calculo el 'knee'
    kneedle = KneeLocator(pers, errs, S=1.0, curve='convex', direction='increasing')
    # Busco el umbral asociado a ese 'knee'
    for i in range(len(pers)):
        if (pers[i] == kneedle.knee):
            opt_thr = thrs[i]
            opt_err = errs[i]

    if(signal_title!=""):
        print("El punto 'knee' es: {}".format(round(kneedle.knee, 2)))
        print("El umbral asociado es: {}".format(opt_thr))
        print("El error asociado al 'knee' es: {}".format(round(opt_err, 2)))

    if(show_n_save):# Imprimo las gráficas
        plt.plot(pers, errs, '-o', linewidth=1)
        plt.vlines(kneedle.knee, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Punto 'knee'")
        plt.xlabel("Porcentaje de descartados")
        plt.ylabel("Error medio")
        plt.legend()
        plt.title("Relación porcentaje de descarte - error para '{}'".format(signal_title))
        plt.gcf().canvas.set_window_title('TFG')
        plt.savefig("results/opt_pers_" + str(N) + "_" + signal_title + ".png")
        plt.show()

        plt.plot(thrs, errs, '-o', linewidth=1)
        plt.vlines(opt_thr, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Umbral del punto 'knee'")
        plt.xlabel("Umbral")
        plt.ylabel("Error medio")
        plt.legend(loc="lower right")
        plt.title("Relación umbral - error para '{}'".format(signal_title))
        plt.gcf().canvas.set_window_title('TFG')
        plt.savefig("results/opt_thrs_" + str(N) + "_" + signal_title + ".png")
        plt.show()

    return opt_thr, kneedle.knee, opt_err

############################
###   OPTIMIZATION DE N  ###
############################

""" Optimización del error medio. Devuelve el punto 'knee', el umbral y error asociado.
- signal_f: señal.
- dom: dominio de la señal.
- N: número de trozos en los que discretizar la señal.
- signal_title: título de la señal.
- show_n_save(op): Flag que indica si mostrar y guardar las gráficas. Por defecto 'True'.
"""
def optimization_N(signal_f, dom, thr, signal_title, show_n_save = True):
    if(signal_title!=""):
        print("\n#####################################################")
        print("    Optimizando N de '{}'".format(signal_title))
        print("#####################################################\n  ")
        print("Umbral fijado: {}.".format(thr))

    Ns = []; errs = []; pers = [];

    for N in range(3,15):
        puntos = np.linspace(dom[0], dom[1], num=2**N)
        signal = np.empty((2**N), dtype=np.float64)
        for i in range(2**N):
            signal[i] = signal_f(puntos[i])
        err, per = experiment_opt(signal, 2**N, thr)
        Ns.append(2**N); errs.append(err); pers.append(per);

    if(signal_title!=""): # Imprimo las listas
        print("Ns:")
        print(Ns)
        print("Errores:")
        print(errs)
        print("Porcentajes de descarte:")
        print(pers)

    # Calculo el 'knee'
    kneedle = KneeLocator(Ns, errs, S=1.0, curve='convex', direction='decreasing')
    # Busco el umbral asociado a ese 'knee'
    for i in range(len(Ns)):
        if (Ns[i] == kneedle.knee):
            opt_err = errs[i]

    if(signal_title!=""):
        print("El punto 'knee' es: {}".format(round(kneedle.knee, 2)))
        print("El error asociado al 'knee' es: {}".format(round(opt_err, 2)))

    if(show_n_save):# Imprimo las gráficas
        plt.plot(Ns, errs, '-o', linewidth=1)
        plt.vlines(kneedle.knee, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Punto 'knee'")
        plt.xlabel("N")
        plt.ylabel("Error medio")
        plt.legend()
        plt.title("Relación N - error para '{}'".format(signal_title))
        plt.gcf().canvas.set_window_title('TFG')
        plt.savefig("results/opt_N_" + str(N) + "_" + str(int(10*thr)) + "_" + signal_title + ".png")
        plt.show()

    return kneedle.knee, opt_err

##################################
###   OPTIMIZATION DE THR Y N  ###
##################################

""" Optimización del error medio. Devuelve el punto 'knee'.
- signal_f: señal.
- dom: dominio de la señal.
- N: número de trozos en los que discretizar la señal.
- signal_title: título de la señal.
"""
def optimization_thr_N(signal_f, dom, signal_title):
    print("\n#####################################################")
    print("    Optimizando umbral y N de '{}'".format(signal_title))
    print("#####################################################\n  ")
    Ns = []; thrs = []; errs = []; pers = [];

    for N in range(3,14):
        thr, per, err = optimization_thr(signal_f, dom, 2**N, "", show_n_save=False)
        Ns.append(2**N); thrs.append(thr); pers.append(per); errs.append(err);

    # Imprimo las listas
    print("Umbrales:")
    print(thrs)
    print("Porcentajes de descarte:")
    print(pers)
    print("Errores:")
    print(errs)

    # Calculo el 'knee'
    kneedle = KneeLocator(Ns, errs, S=1.0, curve='convex', direction='decreasing')
    # Busco el umbral asociado a ese 'knee'
    for i in range(len(Ns)):
        if (Ns[i] == kneedle.knee):
            opt_err = errs[i]
            opt_thr = thrs[i]

    if(signal_title!=""):
        print("El punto 'knee' es: {}".format(round(kneedle.knee, 2)))
        print("El umbral asociado es: {}".format(opt_thr))
        print("El error asociado al 'knee' es: {}".format(round(opt_err, 2)))

    # Imprimo las gráficas
    plt.plot(Ns, errs, '-o', linewidth=1)
    plt.vlines(kneedle.knee, 0, np.amax(np.array(errs)), linestyles='--', colors='g', label="Punto 'knee'")
    plt.xlabel("N")
    plt.ylabel("Error medio")
    plt.legend()
    plt.title("Relación N y umbral - error para '{}'".format(signal_title))
    plt.gcf().canvas.set_window_title('TFG')
    plt.savefig("results/opt_thrN_" + signal_title + ".png")
    plt.show()

    return kneedle.knee, opt_thr, opt_err

#########################
###   ALGUNAS SEÑALES ###
#########################

""" Señal de la suma de seno y coseno.
- x: variable a evaluar.
"""
def sen_plus_cos(x):
    return math.sin(x) + math.cos(x)

""" Señal de la diferencia de seno y coseno.
- x: variable a evaluar.
"""
def sen_minus_cos(x):
    return math.sin(x) - math.cos(x)

""" Señal del producto de seno y coseno.
- x: variable a evaluar.
"""
def sen_plus_cos(x):
    return math.sin(x) * math.cos(x)

""" Señal de 'xsen(x)+xcos(x)'.
- x: variable a evaluar.
"""
def xsen_plus_xcos(x):
    return x*math.sin(x) + x*math.cos(x)

""" Test con señales
"""
def test(func):
    li = np.array([12, 12, 12, 12, 8, 8, 10, 10])
    ha = haar_transform(li,[])
    re = reverse_haar(ha[:1],ha[1:],0)
    print(ha)
    print(re)
    experiment(func, [0, 2*np.pi], 512, thresholding, 0.1, "Señal en [0,2π] (ε=0.1)")
    experiment(func, [0, 2*np.pi], 512, thresholding, 0.5, "Señal en [0,2π] (ε=0.5)")
    experiment(func, [0, 2*np.pi], 512, thresholding, 2, "Señal en [0,2π] (ε=2)")
    input("--- Pulsa 'Enter' para continuar ---\n")


#######################
###       MAIN      ###
#######################

""" Programa principal. """
def main():
    test(xsen_plus_xcos)
    N = 6
    list = [['f', 'Dom(f)', 'N', 'ε', 'Descartes (%)', 'Error medio', 'Factor de compresión'],
         ['sen(x)', '[0,2π]', 512, 0.1],
         ['sen(x)', '[0,2π]', 512, 0.5],
         ['sen(x)', '[0,2π]', 512, 2],
         ['xsen(x)+xcos(x)', '[0,2π]', 512, 0.1],
         ['xsen(x)+xcos(x)', '[0,2π]', 512, 0.5],
         ['xsen(x)+xcos(x)', '[0,2π]', 512, 2]]

    per = np.zeros(N); err = np.zeros(N); fac = np.zeros(N)

    _, per[0], err[0], fac[0] = experiment(math.sin, [0, 2*np.pi], 512, thresholding, 0.1, "sen(x) en [0,2π] (N=512, ε=0.1)")
    _, per[1], err[1], fac[1] = experiment(math.sin, [0, 2*np.pi], 512, thresholding, 0.5, "sen(x) en [0,2π] (N=512, ε=0.5)")
    _, per[2], err[2], fac[2] = experiment(math.sin, [0, 2*np.pi], 512, thresholding, 2, "sen(x) en [0,2π] (N=512, ε=2)")
    _, per[3], err[3], fac[3] = experiment(xsen_plus_xcos, [0, 2*np.pi], 512, thresholding, 0.1, "xsen(x)+xcos(x) en [0,2π] (N=512, ε=0.1)")
    _, per[4], err[4], fac[4] = experiment(xsen_plus_xcos, [0, 2*np.pi], 512, thresholding, 0.5, "xsen(x)+xcos(x) en [0,2π] (N=512, ε=0.5)")
    _, per[5], err[5], fac[5] = experiment(xsen_plus_xcos, [0, 2*np.pi], 512, thresholding, 2, "xsen(x)+xcos(x) en [0,2π] (N=512, ε=2)")

    for k in range(1,N+1):
        list[k].append(per[k-1])
        list[k].append(err[k-1])
        list[k].append(fac[k-1])

    print()
    print(tabulate(list, headers='firstrow', tablefmt='fancy_grid'))

    optimization_thr(math.sin, [0, 2*np.pi], 512, "sen(x) en [0,2π]")
    optimization_thr(xsen_plus_xcos, [0, 2*np.pi], 512, "xsen(x)+xcos(x) en [0,2π]")
    optimization_N(math.sin, [0, 2*np.pi], 0.3, "sen(x) en [0,2π]")
    optimization_N(xsen_plus_xcos, [0, 2*np.pi], 0.3, "xsen(x)+xcos(x) en [0,2π]")
    optimization_thr_N(math.sin, [0, 2*np.pi], "sen(x) en [0,2π]")
    optimization_thr_N(xsen_plus_xcos, [0, 2*np.pi], "xsen(x)+xcos(x) en [0,2π]")


if __name__ == "__main__":
	main()
