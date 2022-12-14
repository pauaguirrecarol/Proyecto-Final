import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
import pandas as pd
import bioread
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.optimize import curve_fit
from scipy import signal #detrend


# -------------------------------------------------------------------------------------------------------------------------------
def FIRrespt(ninput,xinput,nh,h):
    Nx=len(xinput);Nh=len(h);conv=np.convolve(xinput,h);n0=ninput[0]+nh[0];Nconv=Nx+Nh-1;nconv=np.arange(n0,n0 + Nconv)
    R = np.vstack((nconv,conv)).T
    return R   

def FIRrespf(w,h):
    Nh=len(h); H = 0
    for i in np.arange(Nh):
        H = H + h[i]*np.exp(-1j*w*i)
    R = np.vstack((w,abs(H),np.angle(H))).T
    return R 

def IIRrespt(xinput,c,d):
    N1 = len(xinput) # 
    nx = np.arange(-(len(c)-1),N1)
    x = xinput
    for k in np.arange(len(c)-1):
        x = np.insert(x,0,0)
    ny = np.arange(-(len(d)-1),N1)
    y = np.arange(0,len(ny),dtype=np.float64)
    for k in np.arange(len(d)-1):
        y[k] = 0
    for i in np.arange(len(d)-1,len(y)):
        z = 0
        for j in np.arange(1,len(d)):
            z = z - d[j]*y[i-j] + c[j]*x[i-j]
        y[i] = (z + c[0]* x[i])/d[0]
    R = np.vstack((ny,y)).T
    return R

def IIRrespf(w,c,d):
    Nc=len(c); Nd=len(d); Hnum = 0; Hden = 0
    for i in np.arange(Nc):
        Hnum = Hnum + c[i]*np.exp(-1j*w*i)
    for i in np.arange(Nd):
        Hden = Hden + d[i]*np.exp(-1j*w*i)
    H = np.divide(Hnum,Hden)
    R = np.vstack((w,abs(H),np.angle(H))).T
    return R
# -------------------------------------------------------------------------------------------------------------------------------


# FUNCIÓN PARA GRAFICAR SEÑALES     ---------------------------------------------------------------------------------------------
def GraficarOriginalACQ (nombreArchivo):

    # Leer el archivo de datos
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")
    

    canal_elegido = int (input ("Ingrese el canal que desea analizar:"))
    t = ECG.time_index.T

    señal = ECG.channels[canal_elegido].data
    Fs = ECG.channels[canal_elegido].samples_per_second

    # Graficar la señal original
    plt.figure (figsize=(20,4))
    plt.plot (t, señal)
    plt.title ( 'Señal original' )
    plt.xlabel ( 'Muestras' )
    plt.ylabel ( 'Amplitud' )
    plt.grid ()
    plt.savefig('../imagenes/senalGeneradaACQ.jpg', dpi=600, bbox_inches='tight')
    plt.show ()
# -------------------------------------------------------------------------------------------------------------------------------



# FUNCIÓN PARA ESPECTRO     -----------------------------------------------------------------------------------------------------
def Espectro(nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")

    canal_elegido = int (input ("Ingrese el canal que desea analizar:"))
    t = ECG.time_index.T

    señal = ECG.channels[canal_elegido].data

    N = len(señal)
    Fs = ECG.channels[canal_elegido].samples_per_second

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)

    #Graficamos
    plt.figure(figsize=(20,4),dpi=600)
    plt.plot(w*Fs/(2*np.pi),abs(espectro))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Espectro')
    plt.xticks (np.arange (0, Fs/2, 25))
    plt.xlim (0, Fs/2)
    plt.grid(True)
    plt.savefig('../imagenes/espectro.jpg', dpi=600, bbox_inches='tight')
    plt.show()   
# -------------------------------------------------------------------------------------------------------------------------------



# FUNCIÓN PARA FPM     ----------------------------------------------------------------------------------------------------------
def FPM(nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")

    canal_elegido = int (input ("Ingrese el canal que desea analizar:"))

    señal = ECG.channels[canal_elegido].data
    n = np.arange(len(señal))
    
    M = int(input('Ingrese el orden del FPM'))
    
    k = np.arange(0,30,dtype=float)
    FPM = np.piecewise(k,(k>=0)&(k<=M-1),[1/M,0])
    R= FIRrespt(n,señal,k,FPM)
    
    #Grafica de la señal
    plt.figure(figsize=(20,4),dpi=600)
    plt.title('Señal')
    plt.xlabel('n muestras')
    plt.plot(n,señal,"k")
    plt.plot(R[:,0],R[:,1],"y",label=("FPM de Orden",M))
    plt.legend(fontsize=12)
    plt.grid (True)
    plt.savefig('../imagenes/FPM.jpg', dpi=600, bbox_inches='tight')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------------





# FUNCIÓN PARA ELIMINAR TENDENCIAS    --------------------------------------------------------------------------------------------

#Tendencia Lineal
def flineal(x,a,b): #defino la función lineal (a:pendiente - b: ord. al origen)
    return a*x + b

def TendenciaLineal (nombreArchivo):
    #Ingreso la señal
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 
    
    popt, pcov = curve_fit(flineal, n,señal) #realizo el ajuste

    y1 = señal - flineal(n,*popt)

    plt.figure(figsize=(20,4))
    plt.plot(n, señal,label='ECG')
    plt.plot(n,flineal(n,*popt),'k',label='fit lineal')
    plt.plot(n,y1,'r',label='ECG sin tendencia')
    plt.xlabel('n/muestras',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('../imagenes/tendencia_lineal.jpg', dpi=600, bbox_inches='tight')
    plt.show()


    #Tendencia Senoidal
def fsin(x,a,f,c):
    return a*np.sin (2*np.pi*f*x) + c

def TendenciaSenoidal (nombreArchivo):
    #Ingreso la señal
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 
    
    popt1, pcov1 = curve_fit (fsin,n,señal,p0=(0.5,1e-3,0.1)) #realizo el ajuste

    y1 = señal - fsin(n,*popt1)

    plt.figure(figsize=(20,4))
    plt.plot(n, señal,label='ECG')
    plt.plot(n,fsin(n,*popt1),'k',label='fit lineal')
    plt.plot(n,y1,'r',label='ECG sin tendencia')
    plt.xlabel('n/muestras',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('../imagenes/tendencia_senoidal.jpg', dpi=600, bbox_inches='tight')
    plt.show()


#Tendencia Senoidal con detrend
def TendenciaSenoidalDETREND (nombreArchivo):
    #Ingreso la señal
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 
    
    A3 = np.arange(0,N,110) #arreglo
    B3 = tuple(A3) #tupla
    y33 = signal.detrend(señal,bp=A3)


    plt.figure(figsize=(20,4))
    plt.plot(n, señal,label='ECG')
    plt.plot(n,y33,'r',label='ECG sin tendencia')
    plt.xlabel('n/muestras',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('../imagenes/tendencia_senoidal_detrend.jpg', dpi=600, bbox_inches='tight')
    plt.show()


    #Tendencia Exponencial
def fexp (x,a,b,c):
    return a*np.exp(-b*x)+c

def TendenciaExponencial (nombreArchivo):
    #Ingreso la señal
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 
    
    popt2 , pcov2 = curve_fit (fexp,n,señal,p0=(1,1e-3,0.1)) #realizo el ajuste

    y1 = señal - fexp(n,*popt2)

    plt.figure(figsize=(20,4))
    plt.plot(n, señal,label='ECG')
    plt.plot(n,fexp(n,*popt2),'k',label='fit lineal')
    plt.plot(n,y1,'r',label='ECG sin tendencia')
    plt.xlabel('n/muestras',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('../imagenes/tendencia_exponencial.jpg', dpi=600, bbox_inches='tight')
    plt.show()


#Tendencia Exponencial con suavizado con filtro Savitzky-Golay
def TendenciaExponencialFSG (nombreArchivo):
    #Ingreso la señal
    señal = np.loadtxt(nombreArchivo)
    N = len(señal); n = np.arange(N) 

    y_4 = signal.savgol_filter (señal,1080,3) #window size 181, polynomial orden 3 
    y_44 = señal - y_4

    plt.figure(figsize=(20,4))
    plt.plot(n, señal,label='ECG')
    plt.plot(n,y_4,'k',label='y_using_savgol_fit')
    plt.plot(n,y_44,'r',label='ECG sin tendencia')
    plt.xlabel('n/muestras',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('../imagenes/tendencia_exponencial_fsg.jpg', dpi=600, bbox_inches='tight')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------------




# FUNCIÓN FILTROS IIR ----------------------------------------------------------------------------------------------------------
def IIR(nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")

    canal_elegido = int (input ("Ingrese el canal que desea analizar:"))
    t = ECG.time_index.T

    señal = ECG.channels[canal_elegido].data

    N = len(señal)
    Fs = ECG.channels[canal_elegido].samples_per_second

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)

    #Graficamos
    plt.figure(figsize=(20,4),dpi=600)
    plt.plot(w*Fs/(2*np.pi),abs(espectro))
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Espectro')
    plt.xticks (np.arange (0, Fs/2, 25))
    plt.xlim (0, Fs/2)
    plt.grid()
    plt.savefig('../imagenes/filtroiir.jpg', dpi=600, bbox_inches='tight')
    plt.show()
    

    fp1 = int (input ("Ingrese frecuencia de paso inferior del filtro IIR pasa banda: "))
    fp2 = int (input ("Ingrese frecuencia de paso superior del filtro IIR pasa banda: "))
    fr1 = int (input ("Ingrese frecuencia de rechazo inferior del filtro IIR pasa banda: "))
    fr2 = int (input ("Ingrese frecuencia de rechazo superior del filtro IIR pasa banda: "))

    gpass = 1
    gstop = 20
    
    wp = np.array([fp1,fp2])/(Fs/2)
    wr = np.array([fr1,fr2])/(Fs/2)
    
    N3,wc3 = signal.buttord(wp, wr, gpass, gstop, False)
    
    b3, a3 = signal.butter(N3, wc3, 'bandstop')
    w3, h3 = signal.freqz(b3, a3)
    
    # Grafico H(w)
    plt.figure(figsize=(16,4))
    plt.plot(w*Fs/(2*np.pi),abs(espectro))
    plt.plot(w3*Fs/(2*np.pi), np.abs(h3))
    plt.xlabel('f Hz')
    plt.xlim(0,Fs/2)
    plt.grid(True)
    plt.savefig('../imagenes/filtroiir2.jpg', dpi=600, bbox_inches='tight')
    plt.show()
    
    condicion = input('¿continuar? (y/n): ')
    
    while condicion == 'n':
        fp1 = int (input ("Ingrese frecuencia de paso inferior del filtro IIR pasa banda: "))
        fp2 = int (input ("Ingrese frecuencia de paso superior del filtro IIR pasa banda: "))
        fr1 = int (input ("Ingrese frecuencia de rechazo inferior del filtro IIR pasa banda: "))
        fr2 = int (input ("Ingrese frecuencia de rechazo superior del filtro IIR pasa banda: "))

        gpass = 1
        gstop = 20
    
        wp = np.array([fp1,fp2])/(Fs/2)
        wr = np.array([fr1,fr2])/(Fs/2)
    
        N3,wc3 = signal.buttord(wp, wr, gpass, gstop, False)
    
        b3, a3 = signal.butter(N3, wc3, 'bandstop')
        w3, h3 = signal.freqz(b3, a3)
    
    
    # Grafico H(w)
        plt.figure(figsize=(16,4))
        plt.plot(w*Fs/(2*np.pi),abs(espectro))
        plt.plot(w3*Fs/(2*np.pi), np.abs(h3))
        plt.xlabel('f Hz')
        plt.xlim(0,Fs/2)
        plt.grid(True)
        plt.savefig('../imagenes/filtroiirhw.jpg', dpi=600, bbox_inches='tight')
        plt.show()
        condicion = input('¿continuar? (y/n): ')
        
        
    señal_filtrada = IIRrespt(señal,b3,a3)
    plt.figure(figsize=(12,4))
    plt.plot(t,señal,label='ecg con interferencia')
    plt.plot(señal_filtrada[:,0]/Fs,señal_filtrada[:,1],label='ecg filtrada')
    plt.legend(fontsize=12)
    plt.title(('Filtro IIR de orden',N3,''));plt.xlabel('t seg')
    plt.savefig('../imagenes/señal_filtroiir.jpg', dpi=600, bbox_inches='tight')
    plt.show()
# -------------------------------------------------------------------------------------------------------------------------------



