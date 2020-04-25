import cv2
import os
import os
import numpy as np
from imutils import face_utils
import tensorflow as tf
import pickle
import onnx
import dlib
import onnxruntime as ort
from onnx_tf.backend import prepare
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]
def anonymize_face_simple(image, factor=3.0):
	# automatically determine the size of the blurring kernel based
	# on the spatial dimensions of the input image
	(h, w) = image.shape[:2]
	kW = int(w / factor)
	kH = int(h / factor)
	# ensure the width of the kernel is odd
	if kW % 2 == 0:
		kW -= 1
	# ensure the height of the kernel is odd
	if kH % 2 == 0:
		kH -= 1
	# apply a Gaussian blur to the input image using our computed
	# kernel size
	return cv2.GaussianBlur(image, (kW, kH), 0)
def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)
	# return the pixelated blurred image
	return image



class face_re():
    def __init__(self):
        self.knownEncodings = []
        self.knownNames = []
        onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
        self.fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

        self.graph = tf.Graph().as_default()
        self.threshold = 0.63
        with self.graph:
            self.sess = tf.Session()
            # with self.sess:
            saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
            saver.restore(self.sess, 'models/mfn/m1/mfn.ckpt')

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        if os.path.isfile('face_data'):
            data = pickle.loads(open("face_data", "rb").read())
            self.knownEncodings = data["encodings"]
            self.knownNames = data["names"]

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
        """
        Select boxes that contain human faces
        Args:
            width: original image width
            height: original image height
            confidences (N, 2): confidence array
            boxes (N, 4): boxes array in corner-form
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
        Returns:
            boxes (k, 4): an array of boxes kept
            labels (k): an array of labels for each boxes kept
            probs (k): an array of probabilities for each boxes being in corresponding labels
        """
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs,
                                 iou_threshold=iou_threshold,
                                 top_k=top_k,
                                 )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
    def face_embed(self, img):
        # with self.graph:
        #     with self.sess:
        #         print("loading checkpoint ...")
        if len( img.shape) == 3:
            img = img.reshape(1,112,112,3)

        feed_dict = {self.images_placeholder: img, self.phase_train_placeholder: False}
        embeds = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeds
    def prepro_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img
    def face_detect(self, img, w, h):
        confidences, boxes = self.ort_session.run(None, {self.input_name: img})
        boxes, labels, probs = self.predict(w, h, confidences, boxes, 0.7)
        return boxes, labels, probs
    def add_face_2(self, raw_img, name):
        # if (name in self.knownNames):
        #     return False
        h, w, _ = raw_img.shape
        img = self.prepro_img(raw_img)

        boxes, labels, probs = self.face_detect(img, w, h)

        # if face detected
        if boxes.shape[0] > 0:
            x1, y1, x2, y2 = boxes[0, :]
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            aligned_face = self.fa.align(raw_img, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
            aligned_face = cv2.resize(aligned_face, (112, 112))

            # cv2.imwrite(f'faces/tmp/{label}_{frame_count}.jpg', aligned_face)

            aligned_face = aligned_face - 127.5
            aligned_face = aligned_face * 0.0078125
        embeds = self.face_embed(aligned_face)
        for embed in embeds:
            self.knownEncodings.append(embed)
            self.knownNames.append(name)

        return True
    def save_face_data(self):
        data = {"encodings": np.asarray(self.knownEncodings), "names": self.knownNames}
        f = open("face_data", "wb")
        f.write(pickle.dumps(data))
        f.close()
    def add_face(self, img, name):
        if(name in self.knownNames):
            return False
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        if(len(boxes) > 1 or len(boxes) == 0):
            return False
        encoding = face_recognition.face_encodings(rgb, boxes)
        self.knownEncodings.append(encoding)
        self.knownNames.append(name)

        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open("face_data", "wb")
        f.write(pickle.dumps(data))
        f.close()
        return True

    def cover_face(self,img, x1, y1, x2, y2):
        face = img[y1:y2, x1:x2]
        face = anonymize_face_pixelate(face, 20)
        img[y1:y2, x1:x2] = face
        return img
    def draw_rect(self, img, boxes, predictions, cover_face=False):
        for i in range(boxes.shape[0]):
            box = boxes[i, :]

            text = f"{predictions[i]}"

            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 18, 236), 2)
            # Draw a label with a name below the face
            cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)
            if(cover_face):
                if(text == "unknown"):
                    img = self.cover_face(img, x1, y1, x2, y2)

        return img
    def re_face(self, raw_img):
        h, w, _ = raw_img.shape
        img = self.prepro_img(raw_img)
        boxes, labels, probs = self.face_detect(img, w, h)

        faces = []
        boxes[boxes < 0] = 0
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box

            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            aligned_face = self.fa.align(raw_img, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
            aligned_face = cv2.resize(aligned_face, (112, 112))

            aligned_face = aligned_face - 127.5
            aligned_face = aligned_face * 0.0078125

            faces.append(aligned_face)
        if len(faces) > 0:
            predictions = []

            faces = np.array(faces)
            embeds = self.face_embed(faces)
            # feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
            # embeds = sess.run(embeddings, feed_dict=feed_dict)

            # prediciton using distance
            for embedding in embeds:
                diff = np.subtract(np.asarray(self.knownEncodings), embedding)
                dist = np.sum(np.square(diff), 1)
                idx = np.argmin(dist)
                if dist[idx] < self.threshold:
                    predictions.append(self.knownNames[idx])
                else:
                    predictions.append("unknown")


        else:
            return False, [], []
        return True, boxes, predictions
        # for ((top, right, bottom, left), name) in zip(boxes, names):
        #     # draw the predicted face name on the image
        #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        #     y = top - 15 if top - 15 > 15 else top + 15
        #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.75, (0, 255, 0), 2)