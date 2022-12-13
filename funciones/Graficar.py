import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
import pandas as pd

def GraficarOriginal (nombreArchivo):

    # Leer el archivo de datos
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 

    # Graficar la señal original
    plt.figure ()
    plt.plot (n, señal)
    plt.title ( 'Señal original' )
    plt.xlabel ( 'Muestras' )
    plt.ylabel ( 'Amplitud' )
    plt.grid ()
    plt.savefig('imagenes/senalGenerada.jpg', dpi=600, bbox_inches='tight')
    plt.show ()

    