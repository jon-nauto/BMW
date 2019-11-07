# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os
import numpy as np
import cv2
import keras.models
import csv
import json
from keras.models import model_from_json

import tensorflow as tf
#from encrypt_model import AESCipher

import hashlib

from Crypto import Random
from Crypto.Cipher import AES

    
keras.backend.set_image_data_format('channels_first')


class AESCipher(object):

    def __init__(self, _key):
        self.bs = 16
#        print('Using block size = %s' % self.bs)
        self.key = hashlib.sha256(_key.encode()).digest()
#        print('Hash of key="%s" is "%s"' % (_key, self.key))

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
#        print('Iv: "%s"' % iv)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(raw)

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))  # .decode('utf-8')

    def _pad(self, s):
        return s + str.encode((self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs))

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]



KEY = '1234567890'
cypher = AESCipher(KEY)
scale = 0.017
#down up left right cell smoke holding(not using) eyes_closed no_face no_seatbelt(not using)
thresholds_master = np.array([.85,.6,.9,.8,.95,.8,1,.85,.9,1])
agg_frame_data = []

#needed for pyinstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def decrypt_file(input_path):
    if not tf.gfile.Exists(input_path):
        raise Exception('File doesn\'t exist at path: %s' % input_path)
    with open(input_path, 'rb') as file:
        data = file.read()
    decrypt_str = cypher.decrypt(data)
    return decrypt_str

def open_model(): 

    encrypted_json_path = resource_path('models/encrypted/mobilenet_v1-channels_first.json.encrypted')
    if not tf.gfile.Exists(encrypted_json_path):
        print('No file at path: ', encrypted_json_path)
        encrypted_json_path = resource_path('mobilenet_v1-channels_first.json.encrypted')
        print("attempting local folder path:", encrypted_json_path)
    loaded_model_json = decrypt_file(encrypted_json_path)
    
    ############This forces the use of nonencrypted files
#    json_file = open('models/mobilenet_v1-channels_first.json','r')
#    loaded_model_json = json_file.read()
#    json_file.close()
    ############

    loaded_model = model_from_json(loaded_model_json)
    
    output_path = resource_path('mobilenet_v1-channels_first_1.h5')
    
    encrypted_h5_path = resource_path('models/encrypted/mobilenet_v1-channels_first.h5.encrypted')
    if not tf.gfile.Exists(encrypted_h5_path):
        print('No file at path: ', encrypted_h5_path)
        encrypted_h5_path = resource_path('mobilenet_v1-channels_first.h5.encrypted')
        print("attempting local folder path:", encrypted_h5_path)
    h5_file_raw = decrypt_file(encrypted_h5_path)
    with tf.gfile.GFile(output_path, 'wb') as f:
        f.write(h5_file_raw)
        f.close()
	#load weights into new model
    
    ############This forces the use of nonencrypted files
#    output_path = resource_path('models/mobilenet_v1-channels_first.h5')
    ############
    
    
    loaded_model.load_weights(output_path)
    print("Loaded Model from disk")

	#compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    ### If file exists, delete it ##
    if os.path.isfile(output_path):
        os.remove(output_path)
        print("Removed file at path: %s" % output_path)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_path)
    graph = tf.get_default_graph()

    return loaded_model,graph 

def process_frame(predictions, thresholds=thresholds_master):
    if predictions.size != thresholds.size:
        raise Exception("predictions.count=", predictions.count,", but thresholds.count=",thresholds.count)
    results = predictions > thresholds
    frame_data = {"results":results,"predictions":predictions}
#    print("frame_data=",frame_data)
    agg_frame_data.append(frame_data)
    
def export_frame_data(thresholds):
#    print("agg_frame_data=",agg_frame_data)
    path = get_video_source()
    filename = os.path.basename(path)
    base = os.path.splitext(filename)[0]
    csv_path = base + ".csv"
    header = ["Down", "Up", "Left", "Right", "Cell", "Smoking", "Eyes Closed", "No Face"]
    
    with open (csv_path, 'wt', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerow(thresholds)
        for frame_data in agg_frame_data:
            csv_writer.writerow(frame_data["predictions"])

def get_video_source():
    config = 'config.json'
    if tf.gfile.Exists(config):
        with open(config, 'r') as f:
            jsonDict = json.load(f)
            path = jsonDict['file']
            print(jsonDict)
            if not tf.gfile.Exists(path) and path != 'camera':
                path = 'video.mp4'
    else:
        path = 'video.mp4'
    return path        
   
def main():
    
#    testFiles = ["test/000000000001.jpg",
#                 "test/000000000002.jpg",
#                 "test/000000000008.jpg",
#                 "test/000000000009.jpg",
#                 "test/000000000036.jpg",
#                 "test/000000000038.jpg"
#                 ]
    
    print(keras.backend.image_data_format())
    loaded_model,graph = open_model()
    path = get_video_source()
    print("Video Path=", path)
    if path == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)



    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("CUDA is available")
    else:
        print("CUDA is NOT available")

# Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(True):
#    for videoPath in testFiles:#for using test images
    # Capture frame-by-frame
        if cap.isOpened():
            ret, frame = cap.read()
            
            ##########for testing individual images
#            ret = True
#            frame = cv2.imread(resource_path(videoPath))
            ##########
            
            if ret == True:

                if frame.shape[1] < 768:#if w<768 resize.  If w==768 
                    print("IF")
                    frame = cv2.resize(frame,(768,720))
                elif frame.shape[1] != 768:#only adjust shape if size> 768 (ie from HD resolution)
                    print("ELIF")
                    w = int(0.45*frame.shape[1])
                    h = 250
                    frame = frame[h:h+720,w:w+768]
                print("frame.shape=")
                print(frame.shape)
                image = frame
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                frame = frame.astype(np.float64)
                frame = frame.transpose((2,0,1))
                
                print(frame.shape)
                mean = np.array([103.94, 116.78, 123.68])
                mean = mean[:, np.newaxis, np.newaxis]
                frame = frame - mean
                frame *= scale
            
                
                frame = np.expand_dims(frame,4)
                frame = np.moveaxis(frame,-1,0)
                print(frame.shape)
                prediction = loaded_model.predict(frame)
                print ('prediction=', prediction)
                prediction = prediction[0,:,1]
                for i, p in enumerate(prediction):
                    label = ''
#                    down up left right cell smoke holding eyes_closed no_face no_seatbelt
                    if i == 0:label = 'down'
                    if i == 1:label = 'up'
                    if i == 2:label = 'left'
                    if i == 3:label = 'right'
                    if i == 4:label = 'cell'
                    if i == 5:label = 'smoke'
                    if i == 6:label = 'holding'
                    if i == 7:label = 'eyes_closed'
                    if i == 8:label = 'no_face'
                    if i == 9:label = 'no_seatbelt'
                    
                    if p > thresholds_master[i]:
                        print(label,' = TRUE')
                    else:
                        print(label,' = FALSE')
                prediction = np.delete(prediction,9)
                prediction = np.delete(prediction,6)
                print (prediction)
                thresholds = np.delete(thresholds_master,9)
                thresholds = np.delete(thresholds,6)
                print(thresholds)
                
                process_frame(prediction, thresholds)
                cv2.imshow('Image',image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("broken")
                break
        else:
            print("cap is !open")
    export_frame_data(thresholds)
# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
    
