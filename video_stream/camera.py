import cv2
import numpy as np
import tensorflow as tf
import math

# Path to label map file
PATH_TO_LABELS = 'video_stream/labelmap.txt'
# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="video_stream/detect.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

expected_input_shape = (width, height)

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Default camera frame dimensions
imW=1280
imH=780

# Video Reference image
reference_frame = cv2.imread('reference.jpg', cv2.IMREAD_COLOR)
reference_height = reference_frame.shape[0]

def rescale_frame(f):
    frame_height = f.shape[0]
    scaling_factor = reference_height / frame_height

    return cv2.resize(f, None, fx=scaling_factor, fy=scaling_factor)

class VideoCamera(object):
    def __init__(self):
      self.video = cv2.VideoCapture(0)
      
      ret = self.video.set(3,imW)
      ret = self.video.set(4,imH)

    def __del__(self):
      self.video.release()



    def get_frame(self):
      ret, frame = self.video.read()

    #   Preprocess frame
      frame_resized = cv2.resize(frame, (expected_input_shape[1], expected_input_shape[0]))
      frame_normalized = frame_resized.astype(np.float32) / 255.0
      # Add a batch dimension (batch size: 1)
      frame_input = frame_normalized[None, ...]

    #  Perform inference and parse the outputs
      interpreter.set_tensor(input_details[0]['index'], frame_input)
      interpreter.invoke()

      # Get the detection results
      boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
      classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
      scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

      # Loop over all detections and draw detection box if confidence is above minimum threshold
      point_in_video = 0
      for i in range(len(scores)):
          if ((scores[i] > 0.95) and (scores[i] <= 1.0)):
  
              # Get bounding box coordinates and draw box
              ymin = int(max(1,(boxes[i][0] * imH)))
              xmin = int(max(1,(boxes[i][1] * imW)))
              ymax = int(min(imH,(boxes[i][2] * imH)))
              xmax = int(min(imW,(boxes[i][3] * imW)))
             
              cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

              # Draw label
              object_name = labels[int(classes[i])]
              label = '%s: %d%%' % (object_name, int(scores[i]*100))
              labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
              label_ymin = max(ymin, labelSize[1] + 10) 
              cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
              cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

              #   Crop frame
              cropped_frame = frame[ymin:ymax, xmin:xmax]

              # Perform frame matching (template matching) between the camera frame and the reference frame
              ref_frame = reference_frame.copy()
              resized_frame = rescale_frame(cropped_frame)
              result = cv2.matchTemplate(ref_frame, resized_frame, cv2.TM_SQDIFF_NORMED)

              # Find the location (position) of the best match in the result
              min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
              top_left = min_loc
              point_in_video = math.ceil(top_left[0] / 200)

      # Overlay text on frame
      text_to_show = ''
      if point_in_video > 0:
        text_to_show = f"Point in video: {point_in_video}"
      else:
        text_to_show = "Target video not in frame"
      cv2.putText(frame,text_to_show,(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

      ret_frame, jpeg_frame = cv2.imencode('.jpg', frame)
      return jpeg_frame.tobytes()