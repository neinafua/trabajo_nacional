import os
import cv2
import imutils
import tkinter as tk
from tkinter import simpledialog, messagebox

def tomar_fotos(carpeta, num_fotos=200):
    # Verifica si la carpeta existe, si no, la crea
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
        print("Carpeta creada:", carpeta)

    cap = cv2.VideoCapture(0)
    faceclasif = cv2.CascadeClassifier('/home/sistemas/achinti/trabajo_nacional/base| de datos/haarcascade_frontalface_default.xml')
    count = 0

    while count < num_fotos:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar y convertir a escala de grises
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxframe = frame.copy()

        # Detección de caras
        faces = faceclasif.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = auxframe[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            filename = f'rostro_{count}.jpg'
            path = os.path.join(carpeta, filename)
            cv2.imwrite(path, rostro)
            count += 1

        cv2.imshow("RCAC - Capturando Fotos", frame)

        if count >= num_fotos:
            break

        k = cv2.waitKey(1)
        if k == 27:  # Presiona 'Esc' para salir
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Éxito", f"{num_fotos} fotos han sido tomadas y guardadas en '{carpeta}'.")

def iniciar_toma_fotos():
    # Solicita el nombre de la carpeta donde se guardarán las fotos
    nombre_carpeta = simpledialog.askstring("Nombre de Carpeta", "Ingrese el nombre de la carpeta para guardar las fotos:")
    if nombre_carpeta:
        # Carpeta principal donde se guardarán todas las fotos
        ruta_principal = os.path.join(os.getcwd(), "data")
        # Crea la carpeta principal si no existe
        if not os.path.exists(ruta_principal):
            os.makedirs(ruta_principal)

        # Carpeta específica para el usuario
        ruta_carpeta = os.path.join(ruta_principal, nombre_carpeta)
        tomar_fotos(ruta_carpeta)

# Configuración de la ventana principal
ventana = tk.Tk()
ventana.title("Captura de Fotos")

# Botón para iniciar la toma de fotos
boton_iniciar = tk.Button(ventana, text="Tomar Fotos", command=iniciar_toma_fotos)
boton_iniciar.pack(pady=20)

# Iniciar el bucle principal de la interfaz
ventana.mainloop()


