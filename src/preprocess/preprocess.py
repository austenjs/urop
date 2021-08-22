import os

import cv2
import numpy as np
import pandas as pd

def load_data(path_to_csv, path_to_images, img_height, img_width):
  images = []
  classes = []
  rows = pd.read_csv(path_to_csv)

  for i, row in rows.iterrows():
    img_class = row["ClassId"]
    img_path = row["Path"]

    path_to_image = os.path.join(path_to_images, img_path)
    image = cv2.imread(path_to_image)
    image = cv2.resize(image, (img_height, img_width), 3)
    
    if i % 500 == 0:
        print(f"loaded: {i}")
        
    images.append(image)    
    classes.append(img_class)

  X = np.array(images)
  y = np.array(classes)
        
  return (X, y)
