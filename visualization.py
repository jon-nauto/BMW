from queue import Queue
from abc import abstractmethod, ABC
from utils import video_to_frames, file_name_from_index, prediction_per_frame, DataGeneratorInfer,\
    open_model_no_encryption, MAX_FRAME
import argparse
import os
import cv2
import random
import numpy as np
import tensorflow as tf


class CategoryRecord(ABC):
    """
    keep track of logits history for each category according to the `maxsize` manually specified
    """
    def __init__(self, time_range, fps):
        self.maxsize = time_range * fps
        self.logit_record = Queue(maxsize=self.maxsize)

    # update the queue every time we obtain a new logit
    def update_logit_record(self, logit):
        # if record meet maxsize, pop before push the new value
        if self.logit_record.full():
            self.logit_record.get()
        self.logit_record.put(logit)

    def get_logit_record_as_list(self):
        return list(self.logit_record.queue)


class LookingDownCategoryRecord(CategoryRecord):
    category = 'looking-down'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class LookingUpCategoryRecord(CategoryRecord):
    category = 'looking-up'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class LookingLeftCategoryRecord(CategoryRecord):
    category = 'looking-left'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class LookingRightCategoryRecord(CategoryRecord):
    category = 'looking-right'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class CellphoneCategoryRecord(CategoryRecord):
    category = 'cellphone'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class SmokingCategoryRecord(CategoryRecord):
    category = 'smoking'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class HoldObjectCategoryRecord(CategoryRecord):
    category = 'hold-object'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class EyesClosedCategoryRecord(CategoryRecord):
    category = 'eyes-closed'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


class NoFaceCategoryRecord(CategoryRecord):
    category = 'no-face'

    def __init__(self, time_range, fps, threshold):
        super().__init__(time_range, fps)
        self.thresh = threshold


def get_logits_for_all_category(cat_count=8):
    """
    For test purpose:
    A dummy function to generate normalized logits for each category

    Params:
         cat_count: number of categories
    Returns:
         list of logits, one for each category
    """
    generated_logits_each_category = []
    for i in range(cat_count):
        generated_logits_each_category.append(random.uniform(0, 1))

    return generated_logits_each_category


# plot every time we got a new logit ready for each category
def plot_logits(axs, outer_idx, logits_each_category, cat_list, cat_name_list, mark_list, thresh_mark='k-.', fps=1, skip=2):
    """
    setup subplots for all categories

    Params:
        axs: subplot index
        outer_idx: image frame index
        logits_each_category: logit of each category
        cat_list: list of category class instance
        cat_name_list: list of category names, for title
        mark_list: marks and color for each subplot
        thresh_mark: marks and color for threshold
        fps: frame per second, to normalized x-axis of figure

    Returns: None
    """
    for i, (this_logit, this_cat) in enumerate(zip(logits_each_category, cat_list)):
        # update the queue for logits
        this_cat.update_logit_record(this_logit)
        record_length = len(this_cat.get_logit_record_as_list())
        # thresh_list = [this_cat.thresh] * record_length
        # normalize along x axis according to fps, divide by skip to make plot timestamp and that for created video sync
        x_range_normalized = [ci / fps / skip for ci in range(outer_idx + 1 - record_length, outer_idx + 1)]
        x_idx, y_idx = i // 4, i % 4
        # plot logits
        axs[x_idx, y_idx].set_ylim([0, 1])
        axs[x_idx, y_idx].plot(x_range_normalized, this_cat.get_logit_record_as_list(), mark_list[i])
        # plot threshold
        # axs[x_idx, y_idx].plot(x_range_normalized, thresh_list, thresh_mark, label='threshold')
        # axs[x_idx, y_idx].legend(loc="upper right")

        if x_idx == 1:
            axs[x_idx, y_idx].set(xlabel='time / second')
        axs[x_idx, y_idx].set(ylabel='softmax-logit')
        axs[x_idx, y_idx].set_title(cat_name_list[i] + ": frame %d" % outer_idx)


def clear_all_plots(axs, cat_count=8):
    for i in range(cat_count):
        x_idx, y_idx = i // 4, i % 4
        axs[x_idx, y_idx].clear()


def process_all_videos(output_loc, cat_count):
    """
    process extracted video frame by frame

    Params:
        output_loc: location for the extracted frames from the video
        cat_count: number for interested categories
    Returns: numpy array for all logits
    """
    prediction_all_frames = np.zeros((cat_count, MAX_FRAME), dtype=float)
    for idx in range(1, MAX_FRAME):
        if idx % 5 == 0:
            print("##############################################################")
            print("################# processing at frame # %d ################" % idx)
            print("##############################################################")

        image_path = os.path.join(output_loc, file_name_from_index(idx))
        if not os.path.exists(image_path):
            break

        prediction_all_frames[:, idx-1] = prediction_per_frame(image_path)

    return prediction_all_frames


"""
 down up left right cell smoke holding eyes_closed no_face no_seatbelt
"""

cat_index = {'down': 0, 'up': 1, 'left': 2, 'right': 3, 'cell': 4,
             'smoking': 5, 'holding': 6, 'eyes_closed': 7, 'no_face': 8, 'no_seatbelt': 9}


def process_all_videos_generator(output_loc, model_path,
                                 dim=(720, 768), batch_size=1):
    """
        process extracted video frame by frame

        Params:
            output_loc: location for the extracted frames from the video
            dim: desired size for network input
            batch_size: batch size for generator. Please keep to be 1 to avoid unexpected truncation
        Returns: numpy array for all logits
    """
    test_dg = DataGeneratorInfer(output_loc, batch_size=batch_size, dim=dim, n_classes=2, n_channels=3)
    model = open_model_no_encryption(model_path)
    prediction_all_frames_list = model.predict_generator(generator=test_dg, max_queue_size=10, workers=8,
                                                  use_multiprocessing=True)

    # with shape: (n, 10, 2)
    prediction_all_frames_cat = np.asarray(prediction_all_frames_list)
    # extract prediction
    prediction_all_frames = np.squeeze(prediction_all_frames_cat[:, :, 1])
    # remove no-seatbelt
    prediction = np.delete(prediction_all_frames, cat_index['no_seatbelt'], 1)
    # remove looking-up
    logits_prediction = np.delete(prediction, cat_index['up'], 1)

    print("generator shape: ", logits_prediction.shape)
    return logits_prediction


def save_images(location, idx, image_array, prefix='tiled_'):
    if not os.path.exists(location):
        os.mkdir(location)

    image_name = prefix + "%#05d.jpg" % idx
    cv2.imwrite(os.path.join(location, image_name), image_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='preprocessed', nargs='?', choices=['preprocessed', 'live', 'import_csv'],
                        help='a set of mode to choose from')
    parser.add_argument('--plot', default=True, type=bool, help='either to plot the logits')
    parser.add_argument('--input_video_loc', '-i',
                        default='kevin2.mp4',
                        help='input video location')
    parser.add_argument('--output_frames_loc', '-o',
                        default='kevin_tmp2',
                        help='location to find the extracted images and the logits csv file')
    parser.add_argument('--loc_to_save_tiled_image', '-t',
                        help='location to save the tiled image and plots')
    parser.add_argument('--model_path',
                        default='raw_models/mobilenet_v1-channels_first.h5',
                        help='saved model location')
    parser.add_argument('--skip', default=2, type=int, help='show 1 frame every `skip` images')
    parser.add_argument('--gpu_id', '-g', default='3')
    args = parser.parse_args()

    print("##############################################################")
    print("#################### %s mode ##################" % args.mode)
    print("##############################################################")

    if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
        print("CUDA is available")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    else:
        print("CUDA is NOT available")

    # to be tuned
    time_range = 3  # in second
    fps = 4  # frame per second
    cat_count = 8

    # get threshold for each category and remove for 'holding-object' and 'no-seatbelt'
    thresholds = np.array([.85, .6, .9, .8, .95, .8, 1, .85, .9, 1])

    # instantiate for each category
    down = LookingDownCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[0])
    # up = LookingUpCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[1])
    left = LookingLeftCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[2])
    right = LookingRightCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[3])
    cellphone = CellphoneCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[4])
    smoking = SmokingCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[5])
    hold_object = HoldObjectCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[6])
    eyes_closed = EyesClosedCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[7])
    no_face = NoFaceCategoryRecord(time_range=time_range, fps=fps, threshold=thresholds[8])

    cat_list = [down, left, right, cellphone, smoking, hold_object, eyes_closed, no_face]
    cat_name_list = ['looking-down', 'looking-left', 'looking-right', 'cellphone', 'smoking', 'hold-object',
                     'eyes-closed', 'no-face']  # 'looking-up',
    mark_list = ['r-', 'r--', 'g-', 'g--', 'b-', 'b--', 'c-', 'c--']

    # #############################################################
    # ########  extract video to frames and save ##############
    # #############################################################
    if not os.path.exists(args.input_video_loc):
        raise ValueError("video not exist! Please check your `args.input_video_loc`")

    video_to_frames(input_loc=args.input_video_loc, output_loc=args.output_frames_loc)

    # #############################################################
    # ###### preprocess the videos if in 'preprocessed' mode ######
    # #############################################################

    if args.mode == "preprocessed":
        logits_file_name = os.path.join(args.output_frames_loc, 'predicted_logits_all_frames.csv')
        print("logits file name: %s" % logits_file_name)

        if not os.path.exists(logits_file_name):
            print("##############################################################")
            print("wait, preprocessing all videos.........")
            prediction_all_frames = process_all_videos_generator(args.output_frames_loc, args.model_path)
            # prediction_all_frames = process_all_videos(args.output_frames_loc, cat_count=cat_count)
            print("all videos are processed!")
            print("##############################################################")
            np.savetxt(logits_file_name, prediction_all_frames, delimiter=',')

        else:
            prediction_all_frames = np.genfromtxt(logits_file_name, delimiter=',')

        print("shape for all logits: ", prediction_all_frames.shape)
        if prediction_all_frames.shape[1] != cat_count:
            raise ValueError("Please check category numbers")

    elif args.mode == 'import_csv':
        logits_file_name = os.path.join(args.output_frames_loc, 'predicted_logits_all_frames.csv')
        if not os.path.exists(logits_file_name):
            raise ValueError("%s not exist in `import_csv` mode" % logits_file_name)

        prediction_all_frames = np.genfromtxt(logits_file_name, delimiter=',')

        print("shape for all logits: ", prediction_all_frames.shape)
        if prediction_all_frames.shape[1] != cat_count:
            raise ValueError("Please check category numbers")

    # #############################################################
    # ################## save the tiled video #####################
    # #############################################################
    video_out_path = args.input_video_loc.split('.')[0] + '_tiled.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(video_out_path, fourcc, float(fps), (1968, 720))

    # #############################################################
    # #################### iterate for plot #######################
    # #############################################################
    if args.plot:
        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(2, 4)
        fig.set_size_inches(12, 7.2)

    for idx in range(1, MAX_FRAME):
        image_path = os.path.join(args.output_frames_loc, file_name_from_index(idx))
        if not os.path.exists(image_path):
            break

        if idx % args.skip == 0:
            print("##############################################################")
            print("######################## frame %d ########################" % idx)
            print("##############################################################")
        else:
            continue

        if args.mode == 'live':
            logits_each_cat = prediction_per_frame(image_path, args.model_path)
        elif args.mode == 'preprocessed' or args.mode == 'import_csv':
            logits_each_cat = prediction_all_frames[idx-1, :]
        else:
            raise ValueError("mode not recognized!")

        if args.plot:

            plot_logits(axs, idx, logits_each_cat, cat_list, cat_name_list, mark_list, fps=fps, skip=args.skip)

            # show the image
            raw_frame = cv2.imread(image_path)
            # resize image
            frame = DataGeneratorInfer.resize_input_frame(raw_frame)
            # convert plt plot to numpy array
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # tile image and figure horizontally
            combined_frame = np.concatenate((frame, data), axis=1)
            cv2.imshow("driver image vs. predicted logits for each category", combined_frame)
            cv2.waitKey(100)
            clear_all_plots(axs)

            # if args.loc_to_save_tiled_image is None:
            #     raise ValueError('args.loc_to_save_tiled_image should be specified')
            # save_images(args.loc_to_save_tiled_image, idx, combined_frame, prefix='tiled_')

            vw.write(combined_frame)

    vw.release()


if __name__ == '__main__':
    main()
