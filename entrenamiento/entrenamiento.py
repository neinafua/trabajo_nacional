import os 
import cv2
import numpy as np

dataPath = '/home/sistemas/achinti/trabajo_nacional/data'
peopleListe = os.listdir(dataPath)
print('Lista de personas: ', peopleListe)

labels = []
facesData = []
label = 0 

for nameDir in peopleListe:
    personPath =  dataPath + '/' + nameDir
    print('Leyendo imagenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)

        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
    label = label + 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("analizando...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('prueba.xml')
print("modelo guardado")
