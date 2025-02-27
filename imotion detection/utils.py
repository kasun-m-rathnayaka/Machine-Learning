import cv2
import mediapipe as mp


def get_face_landmarks(image, draw=False, static_image_mode=True):
    # read input image
    # convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # create a face mesh object
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode, min_detection_confidence=0.5, max_num_faces=1)
    # process the image
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(image_rgb)
    image_landmarks = []
    # check if there are any faces in the image
    if results.multi_face_landmarks:
        # get the first face
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            mp_drawing.draw_landmarks(image, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
            ls_single_face = results.muti_face_landmarks[0].landmark
            xs = []
            ys = []
            zs = []

            for lm in ls_single_face:
                x, y, z = lm.x, lm.y, lm.z
                xs.append(x)
                ys.append(y)
                zs.append(z)

            for j in range(len(xs)):
                image_landmarks.append(xs[j] - min(xs))
                image_landmarks.append(ys[j] - min(ys))
                image_landmarks.append(zs[j] - min(zs))
        return image_landmarks

    else:
        return None