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
from keras.models import model_from_json
#from keras.preprocessing.image import img_to_array
#from scipy.misc import imread, imresize,imshow
#import matplotlib.pyplot as plt
import tensorflow as tf
#from encrypt_model import AESCipher

import hashlib
#import os
#import sys
#import string
#import random

from Crypto import Random
from Crypto.Cipher import AES

    
keras.backend.set_image_data_format('channels_first')


class AESCipher(object):

    def __init__(self, _key):
        self.bs = 16
        print('Using block size = %s' % self.bs)
        self.key = hashlib.sha256(_key.encode()).digest()
        print('Hash of key="%s" is "%s"' % (_key, self.key))

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        print('Iv: "%s"' % iv)
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
    
    ############
#    json_file = open('models/mobilenet_v1-channels_first.json','r')
#    loaded_model_json = json_file.read()
#    json_file.close()
    ############
#    print('contents of JSON file')
#    print(loaded_model_json)
    loaded_model = model_from_json(loaded_model_json)
    
    output_path = resource_path('models/encrypted/mobilenet_v1-channels_first_test.h5')
    
    encrypted_h5_path = resource_path('models/encrypted/mobilenet_v1-channels_first.h5.encrypted')
    if not tf.gfile.Exists(encrypted_h5_path):
        print('No file at path: ', encrypted_h5_path)
        encrypted_h5_path = resource_path('mobilenet_v1-channels_first.h5.encrypted')
        print("attempting local folder path:", encrypted_h5_path)
    h5_file_raw = decrypt_file(encrypted_h5_path)
    with tf.gfile.GFile(output_path, 'wb') as f:
        f.write(h5_file_raw)
        f.close()
	#load woeights into new model
    
    ############
#    output_path = resource_path('models/mobilenet_v1-channels_first.h5')
    ############
    
    
    loaded_model.load_weights(output_path)
    print("Loaded Model from disk")

	#compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
#    ### If file exists, delete it ##
#    if os.path.isfile(output_path):
#        os.remove(output_path)
#        print("Removed file at path: %s" % output_path)
#    else:    ## Show an error ##
#        print("Error: %s file not found" % output_path)
    graph = tf.get_default_graph()

    return loaded_model,graph 

   
def main():
    
    testFiles = ["test/000000000001.jpg",
#                 "test/000000000002.jpg",
#                 "test/000000000008.jpg",
#                 "test/000000000009.jpg",
#                 "test/000000000036.jpg",
#                 "test/000000000038.jpg"
                 ]
    
    print(keras.backend.image_data_format())
    loaded_model,graph = open_model()
    cap = cv2.VideoCapture(resource_path('video.mp4'))
 #   cap = cv2.VideoCapture(0)

#input_path = 'models/encrypted/mobilenet_v1-channels_first.h5.encrypted'
#output_path = 'models/mobilenet_v1-channels_first_temp.h5'
#
#decrypt_str = decrypt_file(input_path)
#
#with tf.gfile.GFile(output_path, 'wb') as f:
#    f.write(decrypt_str)
#    f.close()



    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("CUDA is available")
    else:
        print("CUDA is NOT available")

# Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

#    while(True):
    for videoPath in testFiles:
    # Capture frame-by-frame
        if cap.isOpened():
#            ret, frame = cap.read()
            
            ret = True
            frame = cv2.imread(resource_path(videoPath))
            image = frame
            if ret == True:

                if frame.shape[1] <= 768:
                    print("IF")
                    frame = cv2.resize(frame,(768,720))
                else:
                    print("ELSE")
                    w = int(0.45*frame.shape[1])
                    h = 250
                    frame = frame[h:h+720,w:w+768]
            #b=b-103.94, g=g-116.78,r=r-123.68
                print("frame.shape=")
                print(frame.shape)
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                frame = frame.astype(np.float64)
                frame = frame.transpose((2,0,1))
                
                print(frame.shape)
#                mean = np.array([103.94, 116.78, 123.68])
                mean = np.array([103.94, 116.78, 123.68])
                mean = mean[:, np.newaxis, np.newaxis]
                frame = frame - mean
                frame *= scale
                print("pre")
#                print(frame[0,:,:])
#                bgr = frame[0,:,:] - meanBGR
#                print('bgr=%s', str(bgr))
#                bgrScaled = np.multiply(bgr, scale,dtype='float32')
#                frame[0,:,:] = bgrScaled 
##            frame[0] = np.subtract(frame[0],meanBGR)
#                print("post")
#                print(frame[0:,:])
            
                
                frame = np.expand_dims(frame,4)
                frame = np.moveaxis(frame,-1,0)
                print(frame.shape)
                prediction = loaded_model.predict(frame)
                print (prediction)
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
                    
                    if p > .7:
                        print(label,' = TRUE')
                    else:
                        print(label,' = FALSE')
                print (prediction)

                cv2.imshow('Image',image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("broken")
                break
        else:
            print("cap is !open")

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
    
