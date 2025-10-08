# Laboratorio 3 - Voz y Frecuencia 
Este laboratorio tiene como objetivo analizar una señal de audio en el dominio del tiempo y de la frecuencia.\
Las librerias que se utilizaron fueron las siguientes:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
```
## **Parte A: – Adquisición de las señales de voz**
Se realiza el siguiente diagrama de flujo:\
<img width="457" height="841" alt="image" src="https://github.com/user-attachments/assets/01ff4809-33bf-4559-83a7-4cc133acb23c" />
<img width="363" height="685" alt="image" src="https://github.com/user-attachments/assets/8ad10946-22a5-4dbc-85e5-2d4403e7b927" />

Primero, se carga un archivo de sonido (.wav) y se grafica su forma de onda, lo que permite observar cómo varía la amplitud del audio a lo largo del tiempo. Luego, mediante la Transformada de Fourier convierte una señal del dominio del tiempo al dominio de la frecuencia, descomposiéndola en sus componentes sinusoidales. Esto permite analizar las frecuencias presentes en la señal, revelando la amplitud y fase de cada frecuencia.
Para hallar las graficarlas en el dominio del tiempo de la voz para los 6 individuos, se tiene el siguiente codigo:
```python
# Archivo de voz .wav
samplerate, data = wavfile.read("/content/drive/MyDrive/Audios lab 3-20251001T191505Z-1-001/Audios lab 3/HOMBRE1.wav")

# Número de muestras
n = data.shape[0]

# Eje de tiempo
t = np.linspace(0, n/samplerate, num=n)

plt.figure(figsize=(10, 4))

# Si es estéreo, graficamos solo un canal (ej: el izquierdo)
if data.ndim > 1:
    plt.plot(t, data[:,0], label="Canal izquierdo")
    plt.plot(t, data[:,1], label="Canal derecho", alpha= 0.7)
    plt.legend()
else:
    plt.plot(t, data, label="Señal mono")
    plt.plot(t, data, "y")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Forma de onda del archivo WAV ")
plt.grid()
plt.show()
```
Luego necesitamos hallar la transformada de Fourier de cada señal y graficar su espectro de magnitudes frecuenciales, con el siguiente codigo:
```python
###FOURIER

# Leer archivo wav
fs, data = wavfile.read("/content/drive/MyDrive/Audios lab 3-20251001T191505Z-1-001/Audios lab 3/HOMBRE1.wav")
# Seleccionar un canal si es estéreo
y = data if data.ndim == 1 else data[:,0]

# FFT
N = len(y)
Y = np.fft.fft(y)
f = np.fft.fftfreq(N, 1/fs)

magnitud = np.abs(Y[:N//2])
espectro = magnitud / np.sum(magnitud)
frecuencias = f[:N//2]

# Graficar espectro (solo frecuencias positivas)
plt.plot(frecuencias, magnitud)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier")
plt.grid()

```
**Caracteristicas de cada señal:** 
a. Frecuencia fundamental.
```python
# Frecuencia fundamental

frec_fund = np.sum(frecuencias * magnitud) / np.sum(magnitud)
print(f"Centroide: {frec_fund:.3f} Hz")
```
b. Frecuencia media.
```python
# Frecuencia media
f_media = np.sum(frecuencias * espectro)
print(f"Frecuencia media: {f_media:.3f} Hz")
```
c. Brillo.
```python
#Calculo del brillo
umbral = 1500  #Energia por encima de 1500 Hz
idx_umbral = np.where(frecuencias >= umbral)[0]
e_total = np.sum(magnitud**2)
e_alta = np.sum(magnitud[idx_umbral]**2)
brillo = e_alta / e_total

print(f"Brillo: {brillo:.3f}")
```
d. Intensidad (energía). 
```python
#Calculo de la Intensidad
intensidad = np.sum(y**2)
print(f"Intensidad: {intensidad:.3f}")
```
**RESULTADOS PARTE A**
- HOMBRE 1:
  
Imagen 1. Señal en dominio del tiempo, Hombre 1\
<img width="991" height="437" alt="image" src="https://github.com/user-attachments/assets/00b82da4-aef7-4444-84a7-4c9523e85b32" />
Imagen 2. Transformada de Fourier y espectro de magnitudes frecuenciales, Hombre 1.\
<img width="632" height="507" alt="image" src="https://github.com/user-attachments/assets/371ccd2e-fbc4-49a3-8145-3f8c2ac31251" />

Imagen 3. Caracteristicas de la señal, Hombre 1.\
<img width="277" height="83" alt="image" src="https://github.com/user-attachments/assets/33af227b-3648-4af3-87f4-7a9c6ed55073" />

  
- HOMBRE 2 :
  
Imagen 4. Señal en dominio del tiempo, Hombre 2\
<img width="990" height="438" alt="image" src="https://github.com/user-attachments/assets/1b1fbea9-7ab7-4685-823c-0c95b66a0e9c" />
Imagen 5. Transformada de Fourier y espectro de magnitudes frecuenciales, Hombre 2.\
<img width="621" height="508" alt="image" src="https://github.com/user-attachments/assets/75b1bd0e-edbd-45c0-bcfe-eb9cd9b7a53d" />

Imagen 6. Caracteristicas de la señal, Hombre 2.\
<img width="261" height="77" alt="image" src="https://github.com/user-attachments/assets/6c183358-6bf3-4d07-915d-699903faae37" />


- HOMBRE 3:
  
Imagen 7. Señal en dominio del tiempo, Hombre 3\
  <img width="987" height="436" alt="image" src="https://github.com/user-attachments/assets/ea86a544-ec4b-4bc5-9e8a-70ded34ea18d" />
Imagen 8. Transformada de Fourier y espectro de magnitudes frecuenciales, Hombre 3.\
  <img width="617" height="505" alt="image" src="https://github.com/user-attachments/assets/f674219c-97cb-4f23-8833-a2d9de42967a" />
  
Imagen 9. Caracteristicas de la señal, Hombre 3.\
  <img width="263" height="76" alt="image" src="https://github.com/user-attachments/assets/531e18c2-7730-4023-a361-a7d78ba2555b" />

- MUJER 1:

Imagen 10. Señal en dominio del tiempo, Mujer 1\
  <img width="987" height="443" alt="image" src="https://github.com/user-attachments/assets/fd0b747d-52cf-4c3c-9411-bcddac26c6ec" />
Imagen 11. Transformada de Fourier y espectro de magnitudes frecuenciales, Mujer 1.\
  <img width="617" height="511" alt="image" src="https://github.com/user-attachments/assets/dba61d90-4375-414e-8d2a-e41e47f4e601" />
  
Imagen 12. Caracteristicas de la señal, Mujer 1.\
  <img width="263" height="82" alt="image" src="https://github.com/user-attachments/assets/3dd95683-c316-4a11-9d43-5688b6afb1d0" />


- MUJER 2:

Imagen 13. Señal en dominio del tiempo, Mujer 2\
  <img width="992" height="438" alt="image" src="https://github.com/user-attachments/assets/8a70deab-3fca-4a75-9a9f-027c6fd3554b" />
Imagen 14. Transformada de Fourier y espectro de magnitudes frecuenciales, Mujer 2.\
  <img width="630" height="505" alt="image" src="https://github.com/user-attachments/assets/226b3471-84fe-4b7d-8dd0-fb91d0750f1f" />

Imagen 15. Caracteristicas de la señal, Mujer 2.\
  <img width="252" height="77" alt="image" src="https://github.com/user-attachments/assets/5b3cf7a2-01de-4772-b359-9dae1b9f964c" />


- MUJER 3:

Imagen 16. Señal en dominio del tiempo, Mujer 3\
  <img width="987" height="437" alt="image" src="https://github.com/user-attachments/assets/d474dc02-2d34-4fb5-b6c1-2cb7875898b5" />
Imagen 17. Transformada de Fourier y espectro de magnitudes frecuenciales, Mujer 3.\
  <img width="636" height="497" alt="image" src="https://github.com/user-attachments/assets/0f32fffe-6dec-4a1f-a312-af1157a94af4" />
  
Imagen 18. Caracteristicas de la señal, Mujer 3.\
  <img width="258" height="82" alt="image" src="https://github.com/user-attachments/assets/1e522bae-9de4-44b5-8a66-633bfd50f5e2" />


## **Parte B: – Medición de Jitter y Shimmer**

Se presenta el diagrama de flujo de la parte B acuerdo al codigo presentado:
<img width="555" height="832" alt="image" src="https://github.com/user-attachments/assets/5d3927d2-ed44-4fea-9519-5939b6f309d3" />

```python

```



## **Parte C: – Comparación y conclusiones **
1. ¿Qué diferencias se observan en la frecuencia fundamental?
   
Las voces masculinas suelen presentar una frecuencia fundamental más baja, generalmente asociada a tonos graves, mientras que las voces femeninas presentan una frecuencia fundamental más alta, lo que produce tonos más agudos.
Esta diferencia se debe principalmente a las características fisiológicas del aparato fonador: los hombres tienen cuerdas vocales más largas y con mayor masa, lo que genera vibraciones más lentas; en cambio, las mujeres tienen cuerdas vocales más cortas y delgadas, produciendo vibraciones más rápidas.
En el análisis espectral, esto se observa claramente porque los picos principales del espectro (correspondientes a la frecuencia fundamental y sus armónicos) se encuentran en frecuencias más bajas para los hombres y más altas para las mujeres.

2. ¿Qué otras diferencias se notan en términos de brillo, media o intensidad?

En cuanto al brillo, las voces femeninas tienden a ser más claras y brillantes debido a una mayor cantidad de energía en las frecuencias altas, lo que les da un timbre más agudo. Por el contrario, las voces masculinas presentan una mayor concentración de energía en frecuencias bajas, lo que genera un timbre más oscuro, profundo y resonante.
Respecto a la media e intensidad, ambas pueden variar según el estilo de habla o la proyección de la voz, pero en general las voces masculinas tienden a tener una energía promedio más estable y uniforme en las bajas frecuencias, mientras que las femeninas pueden mostrar una distribución más amplia de energía en el espectro.
En conjunto, estas diferencias contribuyen a la percepción distintiva entre las voces de hombres y mujeres, tanto en tono como en color y presencia sonora.





  

```python

```
