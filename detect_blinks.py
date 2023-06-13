# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	# розраховуємо евклідову відстань між двома наборами
	# вертикальних критичних точок очей (x, y) - координати
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# розраховуємо евклідому відстань між горизональними
	# критичними точками очей (x, y) - координати
	C = dist.euclidean(eye[0], eye[3])
	
	# розраховуємо співвідношення сторін оче(EAR)
	ear = (A + B) / (2.0 * C)
	
	# повертаємо співвідношення сторін очей
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# створюжмо дві константи, одну для співвідношення сторін очей, щоб
# ідентифікувати кліпання та іншу для обозначення кількості контрольних
# кадрів
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 2
# ініціалізуємо лічільник кадрів та лічільник для всієї кількості лівих та правих моргань
COUNTER_LEFT = 0
COUNTER_RIGHT = 0
TOTAL_LEFT = 0
TOTAL_RIGHT = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# виокремлюємо індекси критичних точок правого та лівого ока
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=1).start()
fileStream = False
time.sleep(1.0)

# проходимо по кадрам з відео стріму
while True:
	# якщо це відео файл, ми повинні подивитись
	# чи не залишилося більше кадрів у буфері для обробки
	if fileStream and not vs.more():
		break
	
	# беремо кадр з потоку, змінюємо його розмір
	# та конвертуємо в відтінки сірого
	frame = vs.read()
	frame = imutils.resize(frame, width=1080)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# розпізнаємо обличчя в сірому кадрі
	rects = detector(gray, 0)
	
    # проходимо крізь детектор обличчя
	for rect in rects:
		# визначаємо критичні точки обличчя, потім
		# окнвертуємо дані точки(x,y - координати) в NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		# виокремлюємо координати правого та лівого ока
		# потім використовуємо координати щоб вирахувати ССО для обох
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		#візуалізуємо область очей
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		# перевіряємо чи ССО нижче порогового значення
		# і якщо це так, збільшуємо лічільник кадрів відповідного ока
		if leftEAR < EYE_AR_THRESH:
			COUNTER_LEFT += 1

		if rightEAR < EYE_AR_THRESH:
			COUNTER_RIGHT += 1	
			
		# коли ССО вище порогового значення
		else:
			# якщо очі були закриті на протязі достатньої кількості кадрів
			# то збільшується лічильник моргань відповідного ока
			if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
				TOTAL_LEFT += 1
				
			if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
				TOTAL_RIGHT += 1

				
			# обнуляжмо лічильники кадрів відповідного ока
			COUNTER_LEFT = 0
			COUNTER_RIGHT = 0
			
        # малюємо на екрані загальну кількість моргань кожного ока
		# та ССО
		cv2.putText(frame, "LEFT BLINKS: {}".format(TOTAL_LEFT), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR LEFT: {:.2f}".format(leftEAR), (200, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		cv2.putText(frame, "RIGHT BLINKS: {}".format(TOTAL_RIGHT), (400, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR RIGHT: {:.2f}".format(rightEAR), (600, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# показ кадру
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()