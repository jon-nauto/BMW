# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:32:50 2019

@author: Jon Emrich
"""

import shutil
import subprocess
import os

def main():
    v_path = 'C:/Users/AlonzoWilkins/Desktop/code/BMW/visualization.py'
    if os.path.exists(v_path):
        os.unlink(v_path)
        print('Deleting file at path: %s' % v_path)
    utils_path = 'C:/Users/AlonzoWilkins/Desktop/code/BMW/utils.py'
    if os.path.exists(utils_path):
        os.unlink(utils_path)
        print('Deleting file at path: %s' % utils_path)
    config_path = 'C:/Users/AlonzoWilkins/Desktop/code/BMW/dist/config.json'
    if os.path.exists(config_path):
        os.unlink(config_path)
        print('Deleting file at path: %s' % config_path)
    videos_path = 'C:/Users/AlonzoWilkins/Desktop/code/BMW/dist/videos'
    if os.path.exists(videos_path):
        shutil.rmtree(videos_path)
        print('Deleting folder at path: %s' % videos_path)             
    print('Copying visualization.py')
    shutil.copy('C:/Users/AlonzoWilkins/Desktop/code/nauto-ai/models/distraction/keras/bmw-demo/visualization.py','C:/Users/AlonzoWilkins/Desktop/code/BMW')
    print('Copying utils.py')
    shutil.copy('C:/Users/AlonzoWilkins/Desktop/code/nauto-ai/models/distraction/keras/bmw-demo/utils.py','C:/Users/AlonzoWilkins/Desktop/code/BMW')
    print('Building EXE')
    cmd = ['pyinstaller','--onefile','BMWDemo_setup.spec']
    subprocess.Popen(cmd,shell=True).wait()
    print('Copying videos')
    shutil.copytree('C:/Users/AlonzoWilkins/Desktop/code/BMW/videos', 'C:/Users/AlonzoWilkins/Desktop/code/BMW/dist/videos')
    print('Copying config.json')
    shutil.copy('C:/Users/AlonzoWilkins/Desktop/code/BMW/config.json','C:/Users/AlonzoWilkins/Desktop/code/BMW/dist/config.json')
    print('Zipping directory')
    shutil.make_archive('zip/BMWDemo','zip','dist')
    print('Complete')
    
if __name__ == '__main__':
    main()