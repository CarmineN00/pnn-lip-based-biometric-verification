import print_landmarks as pl
import cv2

def putFrameCounter(frame,counter):
    # Aggiungi del testo sul frame
    posizione = (20, 50)  # Coordinate della posizione del testo sul frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    scala_font = 2
    colore = (0, 0, 255)  # Colore del testo in formato BGR
    spessore = 2
    cv2.putText(frame, str(counter), posizione, font, scala_font, colore, spessore)
    return frame

if __name__ == "__main__":
    videopath = "Dataset\\Train\\3_1_1_6_25_1.avi"

    video_label = str(''.join(videopath.split("\\")[1].split(".")[0].split("_")[:4]))
    result_name = videopath.split("\\")[2].split(".")[0]

    #result_name = videopath.split("\\")[4].split("_")[0]

    print(result_name)

    # Apri il video
    video = cv2.VideoCapture(videopath)
    larghezza = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    altezza = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    result = cv2.VideoWriter(result_name+"_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (larghezza, altezza))

    counter = 0

    # Loop finché ci sono frame nel video
    while video.isOpened():
        # Leggi il frame corrente
        ret, frame = video.read()
        # Se il frame viene letto correttamente
        if ret:
            # Fai qualcosa con il frame, ad esempio mostralo a schermo
            #frame_result = pl.p_landmarks("",frame,"standard")
            frame_result = pl.p_dynamic_landmarks("",frame)
            if counter == 20:
                cv2.imwrite(result_name+"_result.jpg", frame_result)
            frame_result = putFrameCounter(frame_result,counter)
            '''cv2.imshow("Video", frame_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            result.write(frame_result)
            counter = counter + 1
        else:
            break

    # Rilascia le risorse
    video.release()
    result.release()

