import math

import numpy as np
import os, glob, cv2
import mediapipe as mp
import LipLandmarks as lp


def ottieni_lista_distanze_euclidee(filename):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    cap = cv2.VideoCapture(filename)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 4)

    ret, frame = cap.read()

    landmarks = []

    list_of_euclidean_distances = []

    while ret:

        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        iterator = iter(lp.lip_landmarks)
        for j in range(0, len(lp.lip_landmarks)):
            i = next(iterator)

            # first element of tuple
            point = results.multi_face_landmarks[0].landmark[i[0]]
            node1_x = int(point.x * width)
            node1_y = int(point.y * height)
            landmarks.append((node1_x, node1_y))
            cv2.circle(frame, (node1_x, node1_y), 2, (255, 255, 255), -1)

            # second element of tuple
            point = results.multi_face_landmarks[0].landmark[i[1]]
            node2_x = int(point.x * width)
            node2_y = int(point.y * height)
            landmarks.append((node2_x, node2_y))
            cv2.circle(frame, (node2_x, node2_y), 2, (255, 255, 255), -1)

            # Calcolo della distanza euclidea tra i punti
            d = math.sqrt((node2_x - node1_x) ** 2 + (node2_y - node1_y) ** 2)
            list_of_euclidean_distances.append(d)

    return list_of_euclidean_distances

def main():

    os.chdir("Dataset")
    video_dir = os.listdir()

    for dir in video_dir:
        files = glob.glob(dir + "/*.avi")
        for video in files:
            res = ottieni_lista_distanze_euclidee(video)
            print("Nome del video: ", video)
            print(np.shape(res))
            print(res)
            print("\n\n")
            
if __name__ == "__main__":
    main()