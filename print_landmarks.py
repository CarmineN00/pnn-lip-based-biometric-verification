import cv2
import mediapipe as mp
import LipLandmarks as lp

def p_landmarks(image_path):
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
            cv2.circle(image, (node1_x, node1_y), 1, (255, 255, 255), -1)
            # Secondo elemento della tupla
            point2 = results.multi_face_landmarks[0].landmark[i[1]]
            node2_x = int(point2.x * width)
            node2_y = int(point2.y * height)
            cv2.circle(image, (node2_x, node2_y), 1, (255, 255, 255), -1)

    cv2.imwrite("landmarks_image.jpg", image)


if __name__ == "__main__":
    path = "assets\Mike.jpg"
    p_landmarks(path)