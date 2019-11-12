# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os
#import numpy as np
#import cv2
#import keras.models
#import csv
import json
# insert at 1, 0 is the script path (or '' in REPL)
import tensorflow as tf

#needed for pyinstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
#visualization_path = resource_path("../nauto-ai/models/distraction/keras/bmw-demo/visualization.py")
#if not tf.gfile.Exists(visualization_path):
visualization_path = resource_path("visualization.py")
#sys.path.insert(1,'.')
#else:
#    sys.path.insert(1, resource_path('../nauto-ai/models/distraction/keras/bmw-demo'))

#import visualization
#import subprocess
#import matplotlib


import hashlib

from Crypto import Random
from Crypto.Cipher import AES

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
jsonDict = {}
skip = 0
mode = "preprocessed"
out_path = "output_path"
plot = True
gpu_id = 0



def decrypt_file(input_path):
    if not tf.gfile.Exists(input_path):
        raise Exception('File doesn\'t exist at path: %s' % input_path)
    with open(input_path, 'rb') as file:
        data = file.read()
    decrypt_str = cypher.decrypt(data)
    return decrypt_str

def load_json():
    config = 'config.json'
    if tf.gfile.Exists(config):
        print("file exists")
        with open(config, 'r') as f:
            global skip, plot, gpu_id, jsonDict, mode, out_path
            jsonDict = json.load(f)
            print(jsonDict)
            skip = jsonDict["skip"]
            mode = jsonDict["mode"]
            out_path = jsonDict["output_frames_loc"]
            plot = jsonDict["plot"]
            gpu_id = jsonDict["gpu_id"]
            
    

def get_video_source():
    load_json()
#    print("json=",jsonDict)
    path = jsonDict['file']
    if not tf.gfile.Exists(path):# and path != 'camera':
        path = 'video.mp4'
    
    return path        
   
def main():
    
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
    video_path = get_video_source()
    print("video_path=",video_path)
    call = str("python "+visualization_path+" --skip "+str(skip)+" --mode "+mode+" -i "+video_path+" -o "+out_path+" --model_path "+output_path+" --plot "+str(plot))
    print(call)
    os.system(call)
    print("----------------------------End----------------------------")
#    subprocess.call(call, shell=True)
    #visualization.main(call)
    
    
    ### If file exists, delete it ##
    if os.path.isfile(output_path):
        os.remove(output_path)
    else:    ## Show an error ##
        print("Error: %s file not found" % output_path)
#    while True:
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
    
    
if __name__ == "__main__":
    main()
    
