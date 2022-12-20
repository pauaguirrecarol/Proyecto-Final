import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
import pandas as pd
import bioread
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.optimize import curve_fit
from scipy import signal #detrend
from scipy.signal import find_peaks


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

def vent(vent_name,M):
    nw = np.arange(2*M+1)
    if vent_name == 'rectangular':
        ventana = np.piecewise(nw,nw>=0,[1,0])
    elif vent_name == 'hanning':
        ventana = 0.5*(1+np.cos(2*np.pi*(nw-M)/(2*M+1)))
    elif vent_name == 'hamming':
        ventana = 0.54+0.46*np.cos(2*np.pi*(nw-M)/(2*M+1))
    elif vent_name == 'blackman':
        ventana = 0.42+0.5*np.cos(2*np.pi*(nw-M)/(2*M+1))+0.08*np.cos(4*np.pi*(nw-M)/(2*M+1))
    return np.vstack((nw,ventana)).T

def Hvent(w,ventana):
    z = 0
    for k in np.arange(len(ventana)):
        z = z + ventana[k]*np.exp(-1j*w*k)
    aux = z
    return np.vstack((w,abs(aux),np.angle(aux))).T
# -------------------------------------------------------------------------------------------------------------------------------



def GraficarOriginalACQ (nombreArchivo):

    # Leer el archivo de datos
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)

    # Graficar la señal original
    plt.figure (figsize=(20,4))
    plt.plot (t, señal,"k")
    plt.title ( 'Señal original' )
    plt.xlabel ( 'Tiempo [seg]' )
    plt.ylabel ( 'Amplitud' )
    plt.grid ()
    plt.savefig('imagenes/senalGeneradaACQ.jpg', dpi=600, bbox_inches='tight')
    #plt.show ()

    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal_sin_tendencia = signal.detrend(señal,bp=A3)

    peaks_max = find_peaks (señal_sin_tendencia,distance=500)[0]

    FrecCardiaca = len (peaks_max) * 3

    return FrecCardiaca



def Espectro(nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)

    #Graficamos
    plt.figure(figsize=(20,4),dpi=600)
    plt.plot(w*Fs/(2*np.pi),abs(espectro),"k")
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Espectro')
    plt.xticks (np.arange (0, Fs/2, 25))
    plt.xlim (0, Fs/2)
    plt.grid()
    plt.savefig('imagenes/espectro.jpg', dpi=600, bbox_inches='tight')
    #plt.show()



def FPM (nombreArchivo):
    # Leer el archivo de datos
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)
    n = np.arange (N)



    # Elimino tendencia -----------------------------------------------------------------------------
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal_sin_tendencia = signal.detrend(señal,bp=A3)
    #------------------------------------------------------------------------------------------------



    M = np.arange (2,20,1) #vector de ordenes de los FPM
    k = np.arange (0,30,dtype=np.float64) #vector de muestras de los FPM

    resp = np.empty ((N,len(M)))


    for i,N1 in enumerate (M):
        FPM = np.piecewise (k,(0<=k)&(k<=N1-1),[1/N1,0])
        resp[:,i] = np.convolve (señal_sin_tendencia,FPM,"same")


    peaks_max = find_peaks (señal_sin_tendencia,distance=500)[0]
    peaks_min = find_peaks (-señal_sin_tendencia,distance=500)[0]

    A_max_signal = np.mean (señal_sin_tendencia[peaks_max])
    A_min_signal = np.mean (señal_sin_tendencia[peaks_min])

    aux_max = np.zeros ((len(M),len(peaks_max)))
    aux_min = np.zeros ((len(M),len(peaks_min))) 

    for i in range  (len(M)):
        peaks2_max = find_peaks (resp[:,i],distance=500)[0]
        peaks2_min = find_peaks (-resp[:,i],distance=500)[0]
        aux_max [i,:] = resp [peaks2_max,i]
        aux_min [i,:] = resp [peaks2_min,i]
    A_max = np.mean (aux_max,axis=1) ; A_min = np.mean (aux_min,axis=1)
    A_signal = A_max_signal - A_min_signal ; A_signal_filtr = A_max - A_min
    Atenuac = np.round ((A_signal - A_signal_filtr)*100/A_signal,1) #porcentaje de atencuación


    max_noise = np.zeros (len(M)) ; min_noise = np.zeros (len(M))
    for i in range (len(M)):
        max_noise[i] = max(resp[800:1000,i]) #ruido uniforme en toda la señal
        min_noise[i] = min(resp[800:1000,i])


    A_noise = max_noise - min_noise
    SNR = A_signal_filtr / A_noise
    SNR_dB = 20*np.log10 (A_signal_filtr/A_noise)

    for i in range (len(M)):
        if (Atenuac[i] < 20) & (SNR_dB[i] == max(SNR_dB)):
            print (f"El filtro de orden {M[i]} posee una atenuación de {Atenuac[i]} y una SNR de {SNR_dB[i]} dB.")
            Mejor_FPM = i

    plt.figure(figsize=(20,4),dpi=600)
    plt.title (( 'FPM de orden = ', M[Mejor_FPM] ))
    plt.plot(n, señal_sin_tendencia,color="y")
    plt.plot(n, resp[:,Mejor_FPM],color="k")
    plt.xlabel ("n [muestras]")
    Mejor_Aten = Atenuac[Mejor_FPM]
    Mejor_SNR = SNR_dB[Mejor_FPM]
    plt.grid (True)
    plt.legend (["Señal","FPM"])
    plt.savefig('imagenes/FPM.jpg', dpi=600, bbox_inches='tight')

    return M[Mejor_FPM], Mejor_Aten, Mejor_SNR
    #plot.show()



#Tendencia Senoidal con detrend

def TendenciaSenoidalDETREND (nombreArchivo):
    #Ingreso la señal
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)

    
    A3 = np.arange(0,N,110) #arreglo
    B3 = tuple(A3) #tupla
    y33 = signal.detrend(señal,bp=A3)


    plt.figure(figsize=(20,4),dpi=600)
    plt.plot(t*Fs, señal,"k",label='ECG')
    plt.plot(t*Fs,y33,"y",label='ECG sin tendencia')
    plt.xlabel('n [muestras]',fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig('imagenes/tendencia_senoidal_detrend.jpg', dpi=600, bbox_inches='tight')
    #plt.show()




def FIR(nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")

    canal_elegido = 0
    t = ECG.time_index.T

    señal = ECG.channels[canal_elegido].data

    N = len(señal)
    Fs = ECG.channels[canal_elegido].samples_per_second
    
#................................Elimino tendencia............................................
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal = signal.detrend(señal,bp=A3)
    
#..................................Espectro de Señal.........................................

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)
    eje_f = w*Fs/(2*np.pi)
        
#-.......................Analizo el espectro entre los 40 y 65 Hz............................

    f_min = np.where(eje_f>40)
    f_max = np.where((eje_f>40)&(eje_f<65))
    
    f_m = f_min[0][0] #frecuencia mínima de analisis
    f_M = f_max[0][len(f_max[0])-1] #frecuencia máxima de análisis
    
    #...Determino cual es el valor de mi frecuencia de ruido...    
    vm = np.where(abs(espectro)[f_m:f_M]==np.max(abs(espectro)[f_m:f_M]))
    f_r = eje_f[f_m+vm[0][0]] #frecuencia de ruido
    
#..........................Determino los valores de frecuencias de corte.....................
    
    fc1 = (f_r - 5);    fc2 = (f_r + 5)
    
    wc1 = 2*np.pi*fc1/Fs;    wc2 = 2*np.pi*fc2/Fs
    
#...............................Coeficientes de pasa banda ideal.............................
    n_f = np.arange(-500, 500, dtype=float)
    
    h_bs = np.piecewise(n_f, [n_f == 0],[lambda n_f: 1-(wc2-wc1)/np.pi,
                                         lambda n_f: -(np.sin(wc2*n_f)-np.sin(wc1*n_f))/(np.pi*n_f)])

    ventana_nombre = 'rectangular'
    M = [40,80,200,400] #---Semiancho de las ventanas
    
    
#............Para cada tipo de ventana elijo la mejor señal de acuerdo a la SNR

    Orden = np.zeros(len(M))
    señales = []
    SNRs = np.zeros(len(M))
    Atenuaciones = np.zeros(len(M))
    
    for j in range(len(M)):
        m = M[j]
        vent_BP = vent(ventana_nombre,m)
#................................Desplazo los coeficientes......................................
        n_desp = np.arange(2*m+1,dtype=float)
        h_desp = np.piecewise(n_desp, 
                              [n_desp == m],
                              [lambda n_desp: 1-(wc2-wc1)/np.pi,
                               lambda n_desp: -(np.sin(wc2*(n_desp-m))-np.sin(wc1*(n_desp-m)))/(np.pi*(n_desp-m))])
        Hw2 = 0
        for i in range(len(n_desp)):
            Hw2 = Hw2 + (vent_BP[:,1][i] * h_desp[i]* np.exp(-1j*w*n_desp[i]))
        
            
        señal_filtrada = FIRrespt(t*Fs,señal,n_desp,h_desp)[:,1]
        señal_filtrada=señal_filtrada[m:len(señal_filtrada)-1]
            
#............................Analisis de atenuacion.............................................
        peaks_max_signal = find_peaks(señal,distance=500)[0]
        peaks_min_signal = find_peaks(-señal,distance=500)[0]
        A_max_signal = np.mean(señal[peaks_max_signal])
        A_min_signal = np.mean(señal[peaks_min_signal])

        peaks_max = find_peaks (señal_filtrada,distance=500)[0]
        peaks_min = find_peaks (-señal_filtrada,distance=500)[0]
        A_max = np.mean(señal_filtrada[peaks_max])
        A_min = np.mean(señal_filtrada[peaks_min])


        A_signal = A_max_signal - A_min_signal
        A_signal_filtr = A_max - A_min


        Atenuacion = np.round((A_signal-A_signal_filtr)*100/A_signal,1)


#........................................SNR...................................................
        max_noise = max(señal_filtrada[800:1000])
        min_noise = min(señal_filtrada[800:1000])

        A_noise = max_noise - min_noise
        SNR = np.mean(A_signal_filtr) / A_noise
        SNR_dB = 20*np.log10(SNR)
    
        Orden[j] = m*2 
        señales.append(señal_filtrada)
        SNRs[j] = SNR
        Atenuaciones[j] = Atenuacion
          
#...................Elijo las señales con atenuacion <20%.....................................

    mf = np.where(Atenuaciones<20)[0]
    
    Orden_n = np.zeros(len(mf))
    señales_n = []
    SNRs_n = np.zeros(len(mf))
    Atenuaciones_n = np.zeros(len(mf)) 
    
    for i in range(len(mf)):
        Orden_n[i] = Orden[mf[i]]
        señales_n.append(señales[mf[i]])
        SNRs_n[i] = SNRs[mf[i]]
        Atenuaciones_n[i] = Atenuaciones[mf[i]]
    
#..........De las señales obtenidas, determino la que tiene mayor relacion SNR
 
    mK = np.where(SNRs_n==np.max(SNRs_n))[0][0]
    MEJOR_SEÑAL_Orden = Orden_n[mK]
    MEJOR_SEÑAL_data = señales_n[mK] 
    MEJOR_SEÑAL_SNR = SNRs_n[mK]
    MEJOR_SEÑAL_ATENUACION = Atenuaciones_n[mK]

    MejorSNR = 20*np.log10(MEJOR_SEÑAL_SNR)
    MejorAt = MEJOR_SEÑAL_ATENUACION 

    plt.figure(figsize=(20,4), dpi=600)
    plt.plot(t,señal,'k')
    plt.plot((np.arange(len(MEJOR_SEÑAL_data)))/Fs, MEJOR_SEÑAL_data, 'y')
    plt.legend(('señal original',('Filtrado con Ventana Rectangular orden',MEJOR_SEÑAL_Orden,'')))
    plt.xlabel('Tiempo [seg]', fontsize=12)
    plt.grid(True)
    plt.savefig('imagenes/FIR.jpg', dpi=600, bbox_inches='tight')
    #plt.show()
    
    return MejorSNR, MejorAt




def IIR(nombreArchivo):
#..................................Ingreso Señal..........................................
    
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")
    t = ECG.time_index.T ;     señal = ECG.channels[0].data

    N = len(señal)
    Fs = ECG.channels[0].samples_per_second
    
#..................................Elimino tendencia ......................................
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal = signal.detrend(señal,bp=A3)
    señal = señal.astype(float)
#------------------------------------------------------------------------------------------------
    
#.................................Obtengo espectro de señal....................................

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)
    eje_f = w*Fs/(2*np.pi)
    
#...........................Analizo el espectro entre los 40 y 65 Hz...........................
    f_min = np.where(eje_f>40)
    f_max = np.where((eje_f>40)&(eje_f<65))
    
    f_m = f_min[0][0] #frecuencia mínima de analisis
    f_M = f_max[0][len(f_max[0])-1] #frecuencia máxima de análisis
    
#..................Determino cual es el valor de mi frecuencia de ruido.........................    
    vm = np.where(abs(espectro)[f_m:f_M]==np.max(abs(espectro)[f_m:f_M]))
    f_r = eje_f[f_m+vm[0][0]] #frecuencia de ruido
    
    
#..................Determino los valores de frecuencias de paso y de rechazo...................
    
    fp1 = (f_r - 2);    fp2 = (f_r + 2)
    fr1 = (f_r - 1);    fr2 = (f_r + 1) 
    
#....................Valores de atenuacion de banda de paso y banda de rechazo 
    gp,gstop = 1, 23.5
#..............................................................................

    wp = np.array([fp1,fp2])/(Fs/2)
    ws = np.array([fr1,fr2])/(Fs/2)
    
    system = signal.iirdesign(wp,ws,gp,gstop,analog=False,ftype='butter')
    wz,Hz = signal.freqz(*system,5000) #wz = frecuencias de la señal, Hz = respuesta en frecuencia

    
    
#..................................Filtrado de la señal ................................
    b,a = signal.iirdesign(wp,ws,gp,gstop,analog=False,ftype='butter')
    señal_filtrada = IIRrespt(señal,b,a)
#---------------------------------------------------------------------------------------
 
#..................................Analisis de atenuacion................................

    peaks_max_signal = find_peaks(señal,distance=500)[0]
    peaks_min_signal = find_peaks(-señal,distance=500)[0]
    A_max_signal = señal[peaks_max_signal]
    A_min_signal = señal[peaks_min_signal]

    peaks_max = find_peaks (señal_filtrada[:,1],distance=500)[0]
    peaks_min = find_peaks (-señal_filtrada[:,1],distance=500)[0]
    A_max = señal_filtrada[:,1][peaks_max]
    A_min = señal_filtrada[:,1][peaks_min]
    
    A_signal = A_max_signal - A_min_signal
    A_signal_filtr = A_max - A_min
    
    
    Atenuac = np.arange(len(peaks_max_signal))
    for i in range(len(peaks_max_signal)):
        Atenuac[i] = np.round((A_signal[i]-A_signal_filtr[i])*100/A_signal[i],1)

    Atenuacion = np.mean(Atenuac)

#.......... SNR...............

    max_noise = max(señal_filtrada[:,1][800:1000]) #ruido uniforme en toda la señal
    min_noise = min(señal_filtrada[:,1][800:1000])

    A_noise = max_noise - min_noise
    SNR = np.mean(A_signal_filtr) / A_noise
    SNR_dB = 20*np.log10(SNR)
    

      
    condicion = 1        
    
    while (condicion==1):
              
#.........................amplío la ventana de filtrado................................
        fp1 = fp1-1;        fp2 = fp2+1 

        wp = np.array([fp1,fp2])/(Fs/2)
        ws = np.array([fr1,fr2])/(Fs/2)
    
        system = signal.iirdesign(wp,ws,gp,gstop,analog=False,ftype='butter')
        wz,Hz = signal.freqz(*system,5000)
        
#.......................................Filtro..........................................
        b,a = signal.iirdesign(wp,ws,gp,gstop,analog=False,ftype='butter')
        señal_filtrada_c = IIRrespt(señal,b,a)
#---------------------------------------------------------------------------------------

#..............................Analisis de atenuacion...................................
    
        peaks_max = find_peaks(señal_filtrada_c[:,1],distance=500)[0]
        peaks_min = find_peaks(-señal_filtrada_c[:,1],distance=500)[0]
        
        A_max = señal_filtrada_c[:,1][peaks_max]
        A_min = señal_filtrada_c[:,1][peaks_min]
        
        A_signal_filtr = A_max - A_min

        Atenuac = np.arange(len(peaks_max_signal))
        for i in range(len(peaks_max_signal)):
            Atenuac[i] = np.round((A_signal[i]-A_signal_filtr[i])*100/A_signal[i],1)

        Atenuacion_c = np.mean(Atenuac)

#......................................SNR....................................................
        max_noise = max(señal_filtrada_c[:,1][400:600]) #ruido uniforme en toda la señal
        min_noise = min(señal_filtrada_c[:,1][400:600])

        A_noise = max_noise - min_noise
        SNR_c = np.mean(A_signal_filtr) / A_noise
        SNR_dB_c = 20*np.log10(SNR_c)
       
        
        if SNR_c<SNR:
            condicion = 0
            #.....................Gráfico de señal..........................
            plt.figure(figsize=(20,4),dpi = 600)
            plt.plot(t,señal,label='ECG Original',color='black')
            plt.plot(señal_filtrada[:,0]/Fs,señal_filtrada[:,1],label='ECG-Filtro IIR',color='y')
            plt.legend(fontsize=12)
            plt.title(('Filtro IIR'))
            plt.xlabel('Tiempo [seg]')
            plt.grid (True)
            plt.savefig('imagenes/IIR.jpg', dpi=600, bbox_inches='tight')
            #plt.show()
            MejorAt = Atenuacion
            MejorSNR = SNR
            MejorSNRdB = SNR_dB
            
        else:
            #.......................Gráfico de señal...................................
            señal_filtrada = señal_filtrada_c
            SNR = SNR_c; SNR_dB = SNR_dB_c
            Atenuacion = Atenuacion_c
            condicion = 1      
    print('Filtrado finalizado')
        
    return MejorAt,MejorSNR,MejorSNRdB



def FrecuenciaCardiaca (nombreArchivo):
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)
    n = t*Fs

    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal_sin_tendencia = signal.detrend(señal,bp=A3)

    peaks_max = find_peaks (señal_sin_tendencia,distance=500)[0]

    FrecCardiaca = len (peaks_max) * 3

    return FrecCardiaca