import argparse

from PIL import Image

#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import cv2 as cv
import platform
import collections
import operator
import numpy as np

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

Class = collections.namedtuple('Class', ['id', 'score'])

def input_details(interpreter, key):
  """Returns input details by specified key."""
  return interpreter.get_input_details()[0][key]


def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = input_details(interpreter, 'shape')
  return width, height

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 3)."""
  tensor_index = input_details(interpreter, 'index')
  return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter, dequantize=True):
  """Returns output tensor of classification model.

  Integer output tensor is dequantized by default.

  Args:
    interpreter: tflite.Interpreter;
    dequantize: bool; whether to dequantize integer output tensor.

  Returns:
    Output tensor as numpy array.
  """
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())

  if dequantize and np.issubdtype(output_details['dtype'], np.integer):
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)

  return output_data

def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data

def get_output(interpreter, top_k=1, score_threshold=0.0):
  """Returns no more than top_k classes with score >= score_threshold."""
  scores = output_tensor(interpreter)
  classes = [
      Class(i, scores[i])
      for i in np.argpartition(scores, -top_k)[-top_k:]
      if scores[i] >= score_threshold
  ]
  return sorted(classes, key=operator.itemgetter(1), reverse=True)



def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tf.lite.Interpreter(
      model_path=model_file,
      #experimental_delegates=[
      #    tflite.load_delegate(EDGETPU_SHARED_LIB,
      #                         {'device': device[0]} if device else {})
      #]
        )


def main():
  global imgzi
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  with open("data_labels.txt" , "r") as f:
      data = f.read()
  labels_data = data.split("\n")
  labels = {}
  for x in range(len(labels_data)):
      labels[x] = labels_data[x]



  def predict_image(img):
      interpreter = make_interpreter(args.model)
      interpreter.allocate_tensors()

      size = input_size(interpreter)
      image = img.convert('RGB').resize(size , Image.ANTIALIAS)
      set_input(interpreter , image)

      for _ in range(args.count):
          interpreter.invoke()
          classes = get_output(interpreter , args.top_k , args.threshold)

      return classes

  cap = cv.VideoCapture(0)

  width = 150
  height = 150
  if not cap.isOpened():
      print("Cannot open camera")
      exit()
  while True:
      # 逐帧捕获
      ret , frame = cap.read()
      # 如果正确读取帧，ret为True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      # 我们在框架上的操作到这里

      img = cv.resize(frame , (height , width))
      image = Image.fromarray(cv.cvtColor(img , cv.COLOR_BGR2RGB))
      classes = predict_image(image)
      for klass in classes:
          imgzi = cv.putText(frame , '%s: %.5f' % (labels.get(klass.id , klass.id) , klass.score) , (50 , 300) , cv.FONT_HERSHEY_SIMPLEX , 1.2 , (255 , 255 , 255) , 2)
      cv.imshow('frame' , imgzi)
      if cv.waitKey(1) == ord('q'):
          break
  # 完成所有操作后，释放捕获器
  cap.release()
  cv.destroyAllWindows()


if __name__ == '__main__':
  main()
