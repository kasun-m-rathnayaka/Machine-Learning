import cv2
import argparse
import mediapipe as mp

def process_img(img, face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb).detections

    print(out)
    if out is not None:
        for detection in out:
            detection_data = detection.location_data
            bbox = detection_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # img = cv2.rectangle(img, (x1, y1), (x1+ w, y1 + h), (0, 255, 0), 2)

            # blur faces
            img[y1:y1 + h, x1: x1 + w] = cv2.blur(img[y1:y1 + h, x1: x1 + w], (50, 50))

    return img

# webcam
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='video')
parser.add_argument("--filePath", default='./data/face_anonymized.jpg')
args = parser.parse_args()

# read image
img = cv2.imread("./data/face.jpg")
H, W, _ = img.shape

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == 'image':
        img = process_img(img, face_detection)

        cv2.imshow("Face Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # save image
        cv2.imwrite("./data/face_anonymized.jpg", img)

    elif args.mode == 'video':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter('./data/face_anonymized.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 25.0, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)

            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        output_video.release()
        cv2.destroyAllWindows()