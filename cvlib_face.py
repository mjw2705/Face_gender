import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cvlib as cv
import onnxruntime


def main():
    # gender_model = load_model('gender_detection.model')
    gender_model = onnxruntime.InferenceSession('gender_detection_model.onnx', None)

    classes = ['man','woman']
    frame_shape = (640, 480)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
    cap.set(cv2.CAP_PROP_FPS, 30)

    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.resize(frame, (frame_shape))
        if not success:

            break
        
        face, confidence = cv.detect_face(frame)

        for idx, f in enumerate(face):
            sx, sy, ex, ey = f
            cv2.rectangle(frame, (sx,sy), (ex,ey), (0,255,0), 2)

            face_crop = np.copy(frame[sy:ey,sx:ex])
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)
            # y_pred = gender_model.predict(face_crop)[0]

            input_data = face_crop if isinstance(face_crop, list) else [face_crop]
            feed = dict([(input.name, input_data[n]) for n, input in enumerate(gender_model.get_inputs())])

            y_pred = gender_model.run(None, feed)[0].squeeze()

            # get label with max accuracy
            idx = np.argmax(y_pred)
            label = classes[idx]

            label = "{}: {:.2f}%".format(label, y_pred[idx] * 100)

            Y = sy - 10 if sy - 10 > 10 else sy + 10

            cv2.putText(frame, label, (sx, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.imshow("gender detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()