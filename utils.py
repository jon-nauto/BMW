import cv2
import time
import os
import numpy as np
from keras.models import load_model
import keras


scale = 0.017
MAX_FRAME = 600


class DataGeneratorInfer(keras.utils.Sequence):
    """
    Data generator, can saliently boost the processing speed for inference
    """
    def __init__(self, image_prefix, batch_size=10, dim=(720, 768),
                 n_classes=2, n_channels=3):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.image_prefix = image_prefix
        self.data_list = self._get_data_list()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list) / self.batch_size))

    def _get_data_list(self):
        """get a list of image dirs"""
        data_list = []
        for i in range(1, MAX_FRAME):
            curr_file = self.image_prefix + '/' + file_name_from_index(i)
            if not os.path.exists(curr_file):
                break

            data_list.append(curr_file)
        return data_list

    @staticmethod
    def resize_input_frame(frame):
        # small image
        if frame.shape[1] < 768:
            frame = cv2.resize(frame, (768, 720))
        # image with height 720
        elif frame.shape[1] >= 768 and frame.shape[0] == 720:
            h = frame.shape[1]
            frame = frame[:, h - 768:]
        # HD resolution
        elif frame.shape[1] >= 1632 and frame.shape[0] >= 970:
            w = int(0.45 * frame.shape[1])
            h = 250
            frame = frame[h:h + 720, w:w + 768]

        else:
            raise ValueError("input size not compatible, need further investigation")

        return frame

    @staticmethod
    def preprocess_input(raw_frame):
        frame = DataGeneratorInfer.resize_input_frame(raw_frame)

        # todo: to confirm the operation makes sense
        frame = frame.astype(np.float64)
        frame = frame.transpose((2, 0, 1))

        # print(frame.shape)
        mean = np.array([103.94, 116.78, 123.68])
        mean = mean[:, np.newaxis, np.newaxis]
        frame = frame - mean
        frame *= scale

        frame = np.expand_dims(frame, 4)
        frame = np.moveaxis(frame, -1, 0)

        return frame

    def __getitem__(self, index):
        'Generate one batch of data'
        list_batch = self.data_list[index*self.batch_size: (index+1)*self.batch_size]
        X = np.empty((self.batch_size, self.n_channels, self.dim[0], self.dim[1]))

        down = np.zeros(self.batch_size, dtype=int)
        up = np.zeros(self.batch_size, dtype=int)
        left = np.zeros(self.batch_size, dtype=int)
        right = np.zeros(self.batch_size, dtype=int)
        cellphone = np.zeros(self.batch_size, dtype=int)
        smoke = np.zeros(self.batch_size, dtype=int)
        hold_object = np.zeros(self.batch_size, dtype=int)
        eyes_closed = np.zeros(self.batch_size, dtype=int)
        no_face = np.zeros(self.batch_size, dtype=int)
        no_seatbelt = np.zeros(self.batch_size, dtype=int)

        for i, this_face in enumerate(list_batch):
            img_name = list_batch[i]
            image = cv2.imread(img_name)
            X[i, ] = self.preprocess_input(image)

        print("## batch finished ##")
        down_cat = keras.utils.to_categorical(down, num_classes=self.n_classes)
        up_cat = keras.utils.to_categorical(up, num_classes=self.n_classes)
        left_cat = keras.utils.to_categorical(left, num_classes=self.n_classes)
        right_cat = keras.utils.to_categorical(right, num_classes=self.n_classes)
        cellphone_cat = keras.utils.to_categorical(cellphone, num_classes=self.n_classes)
        smoke_cat = keras.utils.to_categorical(smoke, num_classes=self.n_classes)
        hold_object_cat = keras.utils.to_categorical(hold_object, num_classes=self.n_classes)
        eyes_closed_cat = keras.utils.to_categorical(eyes_closed, num_classes=self.n_classes)
        no_face_cat = keras.utils.to_categorical(no_face, num_classes=self.n_classes)
        no_seatbelt_cat = keras.utils.to_categorical(no_seatbelt, num_classes=self.n_classes)

        return X, [down_cat, up_cat, left_cat, right_cat, cellphone_cat, smoke_cat, hold_object_cat,
                   eyes_closed_cat, no_face_cat, no_seatbelt_cat]


def open_model_no_encryption(model_path):
    """
    Read keras model from h5 file

    Args:
       model_path: location of the source h5 file
    Returns:
       keras model with loaded weights
    """
    
    if not os.path.exists(model_path):
        raise ValueError("model file does not exist!")

    loaded_model = load_model(model_path)
    print("Loaded Model from disk")
    return loaded_model


# extract frames from a video and save
def video_to_frames(input_loc='video.mp4', output_loc='tmp'):
    """
    Function to extract frames from input video file
    and save them as separate frames in an output directory.

    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


def file_name_from_index(idx):
    # generate the file name from the corresponding index
    return "%#05d.jpg" % idx


def prediction_per_frame(image_path, model_path):
    """
    Give logits prediction for each category per frame

    Args:
        image_path: the image to be read from
        model_path: save model path
    Returns:
         numpy array with logits for each category
    """
    if not os.path.exists(model_path):
        raise ValueError("model file does not exist!")

    loaded_model = open_model_no_encryption(model_path)

    raw_frame = cv2.imread(image_path)
    frame = DataGeneratorInfer.preprocess_input(raw_frame)
    # print(frame.shape)
    prediction = loaded_model.predict(frame)
    prediction = prediction[0, :, 1]
    for i, p in enumerate(prediction):
        label = ''
        # down up left right cell smoke holding eyes_closed no_face no_seatbelt
        if i == 0: label = 'down'
        if i == 1: label = 'up'
        if i == 2: label = 'left'
        if i == 3: label = 'right'
        if i == 4: label = 'cell'
        if i == 5: label = 'smoke'
        if i == 6: label = 'holding'
        if i == 7: label = 'eyes_closed'
        if i == 8: label = 'no_face'
        if i == 9: label = 'no_seatbelt'

    # print(prediction)
    prediction = np.delete(prediction, 9) # remove no-seatbelt
    prediction = np.delete(prediction, 1) # remove looking-up

    return prediction
