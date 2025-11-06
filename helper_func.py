from sklearn import preprocessing
import glob
import cv2
import os
import numpy as np


def encode_labels_from_text_to_int(label_arr):
    le = preprocessing.LabelEncoder()
    le.fit(label_arr)
    label_arr_encoded = le.transform(label_arr)
    le.inverse_transform(label_arr)
    return label_arr_encoded











# generator function to generate the dataset for model training
# def generate_dataset(data_path, SIZE):
#     val_images = []
#     val_labels = [] 
#     for directory_path in glob.glob(data_path):
#         label = directory_path.split("\\")[-1]
#         print(label)
#         for img_path in glob.glob(os.path.join(directory_path, "*.png")):
#             # print(img_path)
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
#             img = cv2.resize(img, (SIZE, SIZE))
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             val_images.append(img)
#             val_labels.append(label)
#             images = np.array(val_images)
#             labels = np.array(val_labels)
#     return (images, labels)


def generate_dataset(data_path, SIZE):
    # 1. Initialize lists to store data
    val_images = []
    val_labels = [] 
    
    # Loop over directory paths (e.g., 'training/class_1', 'training/class_2')
    for directory_path in glob.glob(data_path):
        label = directory_path.split("\\")[-1]
        print(f"Processing label: {label}")
        
        # Loop over image files in the current directory
        for img_path in glob.glob(os.path.join(directory_path, "*.png")):
            
            # Load, resize, and convert color (BGR to RGB is often needed for training)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            
            # Safety check in case cv2.imread fails
            if img is None:
                continue

            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # or cv2.COLOR_BGR2RGB if that was the intent

            # Append the data to the efficient Python lists
            val_images.append(img)
            val_labels.append(label)
            
    # 2. FIX: Convert the lists to NumPy arrays ONCE after all loops finish.
    # This is fast and guarantees 'images' and 'labels' are defined.
    images = np.array(val_images)
    labels = np.array(val_labels)
    
    return (images, labels)