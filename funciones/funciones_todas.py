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

    # Graficar la señal original
    plt.figure (figsize=(20,4))
    plt.plot (t, señal,"k")
    plt.title ( 'Señal original' )
    plt.xlabel ( 'Muestras' )
    plt.ylabel ( 'Amplitud' )
    plt.grid ()
    plt.savefig('imagenes/senalGeneradaACQ.jpg', dpi=600, bbox_inches='tight')
    #plt.show ()



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



def FPM(nombreArchivo):
    # Leer el archivo de datos
    ECG = bioread.read_file (nombreArchivo)
    t = ECG.time_index.T

    señal = ECG.channels[0].data
    Fs = ECG.channels[0].samples_per_second
    N = len(señal)
    

    # Elimino tendencia -----------------------------------------------------------------------------
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal_sin_tendencia = signal.detrend(señal,bp=A3)
    #------------------------------------------------------------------------------------------------


    M = np.arange (5,25,5) #vector de ordenes de los FPM
    k = np.arange (0,30,dtype=np.float64)

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
        max_noise[i] = max(resp[400:600,i]) #ruido uniforme en toda la señal
        min_noise[i] = min(resp[400:600,i])


    A_noise = max_noise - min_noise
    SNR = A_signal_filtr / A_noise
    SNR_dB = 20*np.log (A_signal_filtr/A_noise)
        
    for i in range (len(M)):
        if (Atenuac[i] < 20)&(SNR_dB[i] == max(SNR_dB)):
            print (f"El filtro de orden {M[i]} posee una atenuación de {Atenuac[i]} y una SNR de {SNR_dB[i]} dB.")
            Mejor_FPM = i

    plt.figure(figsize=(20,4))
    plt.plot(t*Fs, señal_sin_tendencia,color="y")
    plt.plot(t*Fs, resp[:,Mejor_FPM],color="k")
    plt.title (( 'FPM de orden = ', M[Mejor_FPM] ))
    plt.grid (True)
    plt.savefig('imagenes/FPM.jpg', dpi=600, bbox_inches='tight')
    #plt.show()



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
    t = ECG.time_index.T

    señal = ECG.channels[0].data

    N = len(señal)
    Fs = ECG.channels[0].samples_per_second
    
    # Elimino tendencia -----------------------------------------------------------------------------
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal = signal.detrend(señal,bp=A3)
    #------------------------------------------------------------------------------------------------


    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)
    eje_f = w*Fs/(2*np.pi)

        
#------Analizo el espectro entre los 40 y 65 Hz
    f_min = np.where(eje_f>40)
    f_max = np.where((eje_f>40)&(eje_f<65))
    
    f_m = f_min[0][0] #frecuencia mínima de analisis
    f_M = f_max[0][len(f_max[0])-1] #frecuencia máxima de análisis
    
    #...Determino cual es el valor de mi frecuencia de ruido...    
    vm = np.where(abs(espectro)[f_m:f_M]==np.max(abs(espectro)[f_m:f_M]))
    f_r = eje_f[f_m+vm[0][0]] #frecuencia de ruido
    
    #...Determino los valores de frecuencias de corte
    
    fc1 = (f_r - 5);    fc2 = (f_r + 5)
    
    wc1 = 2*np.pi*fc1/Fs;    wc2 = 2*np.pi*fc2/Fs
    
#...Coeficientes de pasa banda ideal.....
    n_f = np.arange(-500, 500, dtype=float)
    
    h_bs = np.piecewise(n_f, [n_f == 0],[lambda n_f: 1-(wc2-wc1)/np.pi,
                                         lambda n_f: -(np.sin(wc2*n_f)-np.sin(wc1*n_f))/(np.pi*n_f)])

    ventana_nombre = 'rectangular'
    M = [40,80,200,400] #---Semiancho de las ventanas
    
    Mejores_S = []; Mejor_SNR = []
    #Para cada tipo de ventana elijo la mejor señal de acuerdo a la SNR
    señales = []
    SNRs = []
    for j in range(len(M)):
        m = M[j]
        vent_BP = vent(ventana_nombre,m)
#......desplazo los coeficientes
        n_desp = np.arange(2*m+1,dtype=float)
        h_desp = np.piecewise(n_desp, [n_desp == m],[lambda n_desp: 1-(wc2-wc1)/np.pi,
                                             lambda n_desp: -(np.sin(wc2*(n_desp-m))-np.sin(wc1*(n_desp-m)))/(np.pi*(n_desp-m))])
        Hw2 = 0
        for i in range(len(n_desp)):
            Hw2 = Hw2 + (vent_BP[:,1][i] * h_desp[i]* np.exp(-1j*w*n_desp[i]))
            
        señal_filtrada = signal.lfilter(vent_BP[:,1]*h_desp, 1,señal)
            
        #Analisis de atenuacion
        peaks_max_signal = find_peaks(señal,distance=500)[0];peaks_min_signal = find_peaks(-señal,distance=500)[0]
        A_max_signal = np.mean(señal[peaks_max_signal]) ; A_min_signal = np.mean(señal[peaks_min_signal])

        peaks_max = find_peaks(señal_filtrada,distance=500)[0]
        peaks_min = find_peaks(-señal_filtrada,distance=500)[0]
        A_max = np.mean(señal_filtrada[peaks_max]) ; A_min = np.mean(señal_filtrada[peaks_min])

        A_signal = A_max_signal - A_min_signal ; A_signal_filtr = A_max - A_min
        Atenuac = np.round (abs((A_signal - A_signal_filtr)*100/A_signal),1)

        ### Ruido
        max_noise = max(señal_filtrada[0:200]) #ruido uniforme en toda la señal
        min_noise = min(señal_filtrada[0:200])
        A_noise = max_noise - min_noise
        SNR = A_signal_filtr / A_noise
        SNR_dB = 20*np.log (A_signal_filtr/A_noise)
            
        señales.append([ventana_nombre,m*2,señal_filtrada])
        SNRs.append([ventana_nombre,m*2,SNR,SNR_dB,Atenuac])
          
    #Elijo las señales con atenuacion <20%
    valores_atenuac = np.zeros(len(M))
    for k in range(len(M)):
        valores_atenuac[k]= SNRs[k][4]            
    mf = np.where((valores_atenuac<20))
    
    L1_snr = []; L2_signal = []
    for i in range(len(mf[0])):
        L1_snr.append(SNRs[i])
        L2_signal.append(señales[i])
            
    Mejores_S.append(L2_signal); Mejor_SNR.append(L1_snr)
    
    #De las señales obtenidas, determino la que tiene mayor relacion SNR
    valores_SNR = []
    for i in range(len(Mejor_SNR[0])):   
        valores_SNR.append(Mejor_SNR[0][i][2])
        
    mK = np.where(valores_SNR==np.max(valores_SNR))
    MEJOR_SEÑAL_data = Mejores_S[0][mK[0][0]] 
    MEJOR_SEÑAL_SNR = Mejor_SNR[0][mK[0][0]]

   
    plt.figure(figsize=(20,4),dpi=600)
    plt.plot(t,señal,'k')
    plt.plot(t, MEJOR_SEÑAL_data[2], 'y')
    plt.legend(('señal original',('Filtrado con ',MEJOR_SEÑAL_data[0],'orden',MEJOR_SEÑAL_data[1],'')))
    plt.xlabel('t [seg]', fontsize=12)
    plt.grid (True)
    plt.savefig('imagenes/FIR.jpg', dpi=600, bbox_inches='tight')
    #plt.show()
    
    #print('SNR:', MEJOR_SEÑAL_SNR[3],'dB')
    #print('Atenuacion:', MEJOR_SEÑAL_SNR[4],'%')




def IIR(nombreArchivo):
    #...Ingreso Señal...
    
    ECG = bioread.read_file (nombreArchivo)
    canales = ECG.channels
    for i in np.arange (0,len(canales)):
        print (f"Canal {i}: {canales[i]}")

    canal_elegido = 0
    t = ECG.time_index.T

    señal = ECG.channels[canal_elegido].data

    N = len(señal)
    Fs = ECG.channels[canal_elegido].samples_per_second
    
    # Elimino tendencia -----------------------------------------------------------------------------
    A3 = np.arange(0,N,250) #arreglo
    B3 = tuple(A3) #tupla
    señal = signal.detrend(señal,bp=A3)
    #------------------------------------------------------------------------------------------------
    
    #...Obtengo espectro de señal

    espectro = fft(señal)/N
    w = np.linspace(0,2*np.pi,N)
    eje_f = w*Fs/(2*np.pi)
    
    #---Analizo el espectro entre los 40 y 65 Hz
    f_min = np.where(eje_f>40)
    f_max = np.where((eje_f>40)&(eje_f<65))
    
    f_m = f_min[0][0] #frecuencia mínima de analisis
    f_M = f_max[0][len(f_max[0])-1] #frecuencia máxima de análisis
    
    #...Determino cual es el valor de mi frecuencia de ruido...    
    vm = np.where(abs(espectro)[f_m:f_M]==np.max(abs(espectro)[f_m:f_M]))
    f_r = eje_f[f_m+vm[0][0]] #frecuencia de ruido
    
    i = 5;j = 2
    
    #...Determino los valores de frecuencias de paso y de rechazo
    
    fp1 = (f_r - i);    fp2 = (f_r + i)
    fr1 = (f_r - j);    fr2 = (f_r + j) 
    
    #...Valores de atenuacion de banda de paso y banda de rechazo 
    gpass = 1;     gstop = 20
    
    wp = np.array([fp1,fp2])/(Fs/2)
    wr = np.array([fr1,fr2])/(Fs/2)
    
    N3,wc3 = signal.buttord(wp, wr, gpass, gstop, False)
    
    b3, a3 = signal.butter(N3, wc3, 'bandstop')
    w3, h3 = signal.freqz(b3, a3)
    
    #...Filtrado de la señal  
    señal_filtrada = IIRrespt(señal,b3,a3)
    
    #Analisis de atenuacion: Saco los picos en señal original y señal filtrada
    peaks_max = find_peaks (señal,distance=500)[0]
    peaks_min = find_peaks (-señal,distance=500)[0]
    A_max_signal = np.mean(señal[peaks_max]) ; A_min_signal = np.mean(señal[peaks_min])

    peaks_max = find_peaks (señal_filtrada[:,1],distance=500)[0]
    peaks_min = find_peaks (-señal_filtrada[:,1],distance=500)[0]
    A_max = np.mean(señal_filtrada[:,1][peaks_max]) ; A_min = np.mean(señal_filtrada[:,1][peaks_min])
    
    A_signal = A_max_signal - A_min_signal ; A_signal_filtr = A_max - A_min

    Atenuac = np.round ((A_signal - A_signal_filtr)*100/A_signal,1)

    ### Ruido
    max_noise = max(señal_filtrada[:,1][400:600]) #ruido uniforme en toda la señal
    min_noise = min(señal_filtrada[:,1][400:600])

    A_noise = max_noise - min_noise
    SNR = A_signal_filtr / A_noise
    SNR_dB = 20*np.log (A_signal_filtr/A_noise)
      
    condicion = 1        
    
    while (condicion==1):
        
        i = i+1;     j = j+1    
        #amplío la ventana de filtrado
        fp1 = (f_r - i);        fp2 = (f_r + i)
        fr1 = (f_r - j);        fr2 = (f_r + j) 

        wp = np.array([fp1,fp2])/(Fs/2);        wr = np.array([fr1,fr2])/(Fs/2)

        N3_c,wc3 = signal.buttord(wp, wr, gpass, gstop, False)

        b3, a3 = signal.butter(N3_c, wc3, 'bandstop')
        w3, h3 = signal.freqz(b3, a3)
        
        #...Filtro
        señal_filtrada_c = IIRrespt(señal,b3,a3)
        
         #Analisis de atenuacion
    ### Saco los picos en señal original y señal filtrada
    
        peaks_max = find_peaks (señal_filtrada_c[:,1],distance=500)[0]
        peaks_min = find_peaks (-señal_filtrada_c[:,1],distance=500)[0]
        A_max = np.mean(señal_filtrada_c[:,1][peaks_max]) ; A_min = np.mean(señal_filtrada_c[:,1][peaks_min])
        
        A_signal = A_max_signal - A_min_signal ; A_signal_filtr = A_max - A_min


        Atenuac_c = np.round ((A_signal - A_signal_filtr)*100/A_signal,1)

        ### Ruido
        max_noise = max(señal_filtrada_c[:,1][400:600]) #ruido uniforme en toda la señal
        min_noise = min(señal_filtrada_c[:,1][400:600])

        A_noise = max_noise - min_noise
        SNR_c = A_signal_filtr / A_noise
        SNR_dB_c = 20*np.log (A_signal_filtr/A_noise)
       
        
        if SNR_c<SNR:
            condicion = 0
            #...Gráfico de señal
            plt.figure(figsize=(20,4),dpi=600)
            plt.plot(t,señal,"k",label='ecg con interferencia')
            plt.plot(señal_filtrada[:,0]/Fs,señal_filtrada[:,1],"y",label='ecg filtrada')
            plt.legend(fontsize=12)
            plt.title(('Filtro IIR de orden',N3,''));plt.xlabel('t [seg]')
            plt.grid (True)
            plt.savefig('imagenes/IIR.jpg', dpi=600, bbox_inches='tight')
            #plt.show()


            print("ATENUACION:",Atenuac)
            print('SNR:',SNR)
            print('SNR(dB):',SNR_dB)
            
        else:
            #...Gráfico de señal      
            señal_filtrada = señal_filtrada_c
            SNR = SNR_c; SNR_dB = SNR_dB_c        
            N3 = N3_c
            condicion = 1      
    print('Filtrado finalizado')
        
        