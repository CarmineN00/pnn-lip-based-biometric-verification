import cv2
import mediapipe as mp
import LipLandmarks as lp
from scipy.spatial import Delaunay
import numpy as np
import math 


def p_landmarks(image_path,type):
    landmarks = []

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    results = face_mesh.process(image)

    iterator = iter(lp.lip_landmarks)
    for j in range(0, len(lp.lip_landmarks)):
        i = next(iterator)

        if results and results.multi_face_landmarks:
        # Primo elemento della tupla
            point1 = results.multi_face_landmarks[0].landmark[i[0]]
            node1_x = int(point1.x * width)
            node1_y = int(point1.y * height)
            if(type=="delaunay"):
                landmarks.append((node1_x,node1_y))
            elif(type=="standard"):
                cv2.circle(image, (node1_x, node1_y), 2, (255, 0, 0), -1)
            # Secondo elemento della tupla
            point2 = results.multi_face_landmarks[0].landmark[i[1]]
            node2_x = int(point2.x * width)
            node2_y = int(point2.y * height)
            if(type=="delaunay"):
                landmarks.append((node2_x,node2_y))
            elif(type=="standard"):
                cv2.circle(image, (node2_x, node2_y), 2, (255, 0, 0), -1)
                cv2.line(image, (node1_x, node1_y), (node2_x,node2_y), (255, 255, 255), 1)

    if(type=="standard"):
        cv2.imwrite("standard_landmarks.jpg", image)
    elif(type=="delaunay"):
        landmarks = set(landmarks)
        landmarks = list(landmarks)

        landmarks_np = np.array(landmarks)
        triangles = Delaunay(landmarks_np)
        distances =[]

        # Set per tenere traccia dei segmenti unici
        unique_segments = set()

        # Lista per salvare le distanze euclidee
        distances = []

        # Itera sui triangoli
        for tri in triangles.simplices:
            pt1, pt2, pt3 = landmarks_np[tri]
            
            # Controllo per segmenti unici
            segment1 = tuple(sorted([tuple(pt1), tuple(pt2)]))
            segment2 = tuple(sorted([tuple(pt2), tuple(pt3)]))
            segment3 = tuple(sorted([tuple(pt3), tuple(pt1)]))
            
            # Disegna solo segmenti unici
            if segment1 not in unique_segments:
                cv2.circle(image,(pt1[0],pt1[1]), 2, (255, 0, 0), -1)
                cv2.line(image, tuple(pt1), tuple(pt2), (255, 255, 255), 1)
            if segment2 not in unique_segments:
                cv2.circle(image,(pt2[0],pt2[1]), 2, (255, 0, 0), -1)
                cv2.line(image, tuple(pt2), tuple(pt3), (255, 255, 255), 1)
            if segment3 not in unique_segments:
                cv2.circle(image,(pt3[0],pt3[1]), 2, (255, 0, 0), -1)
                cv2.line(image, tuple(pt3), tuple(pt1), (255, 255, 255), 1)

        cv2.imwrite("delaunay_landmarks.jpg", image)
        # Stampa il numero di distanze euclidee ottenute (41)
        # print("Distanze euclidee:")
        # print(len(distances))
        

if __name__ == "__main__":
    path = "assets\Mike.jpg"

    # il secondo paramentro deve essere : "standard" o "delaunay"
    p_landmarks(path,"delaunay")