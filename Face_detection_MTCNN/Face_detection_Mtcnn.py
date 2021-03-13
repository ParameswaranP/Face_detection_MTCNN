from mtcnn import MTCNN
import cv2
detector = MTCNN()


def draw(image, box_coordinates, img_confidence):
    x, y, w, h = box_coordinates
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(image, str(int(img_confidence*100)) + "%", (x, y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2)
    # Coordinates that are coming out from mtcnn are in yolo format so we needed add x+w and y+h in pt2
    return image


img = cv2.cvtColor(cv2.imread(r"./thor_face.jpg"), cv2.COLOR_BGR2RGB)
# MTCNN will work well with RGB but cv2 reads image in BGR so it is advisable to cvt BGR2RGB
img = cv2.resize(img, (0, 0), img, 0.5, 0.5)
detection = detector.detect_faces(img)
# detector return format
# [
#     {
#         'box': [277, 90, 48, 63],
#         'keypoints':
#         {
#             'nose': (303, 131),
#             'mouth_right': (313, 141),
#             'right_eye': (314, 114),
#             'left_eye': (291, 117),
#             'mouth_left': (296, 143)
#         },
#         'confidence': 0.99851983785629272
#     }
# ]
coordinates = detection[0]["box"]
conf = detection[0]["confidence"]
img = draw(img, coordinates, conf)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# For good color visualization we needed to reconvert RBG2BGR
cv2.imshow("Hello", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
