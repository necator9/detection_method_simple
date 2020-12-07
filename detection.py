# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Detection algorithm. All the stages of the detection algorithm are called from this module.

import cv2
import numpy as np
import queue
import timeit
from collections import deque
from functools import wraps
import logging

from pre_processing import PreprocessImg
import saver
from sl_connect import SlAppConnSensor

logger = logging.getLogger('detect.detect')


class Detection(object):
    def __init__(self, stop_ev, orig_img_q, config):
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.config = config

        self.frame = Frame(config)
        self.mean_tracker = MeanResultTracker(*config['lamp_on_criteria'])

        self.empty = np.empty([0])

        self.time_measurements = list()
        self.time_window = config['time_window']
        self.sl_app_conn = SlAppConnSensor(config['sl_conn']['detect_port'], [config['sl_conn']['sl_port']])
        self.pre_processing = PreprocessImg(config)

        if config['save_csv']:
            self.save_csv = saver.SaveCSV(config['out_dir'])
        if config['save_img'] or config['stream']['enabled']:
            self.save_img = saver.SaveImg(config)

        if any([config['save_csv'], config['save_img'], config['save_img'], config['stream']['enabled']]):
            self.save_flag = True
        else:
            self.save_flag = False

    @staticmethod
    def prepare_array_to_save(data, img_num, av_bin_result, lamp_status):
        # Add image number and row indices as first two columns to distinguish objects later
        return np.column_stack((np.full(data.shape[0], img_num), np.arange(data.shape[0]), data,
                                np.full(data.shape[0], av_bin_result), np.full(data.shape[0], lamp_status)))

    def run(self):
        logger.info("Detection has started")
        steps = dict()

        iterator = 0
        lamp_status = False
        while not self.stop_event.is_set():
            start_time = timeit.default_timer()

            try:
                orig_img = self.orig_img_q.get(timeout=2)

                lamp_event = self.sl_app_conn.check_lamp_status()
                if lamp_event:
                    lamp_status = not lamp_status
                    logger.debug("Skipping the current frame due to the lamp event")
                    self.orig_img_q.get(timeout=2)  # Blank call to skip current frame
                    logger.debug("Recapturing the frame")
                    orig_img = self.orig_img_q.get(timeout=2)  # Recapture image

            except queue.Empty:
                logger.warning("Timeout reached, no items can be received from orig_img_q")
                continue

            steps['resized_orig'], steps['mask'], steps['filtered'], steps['filled'] = \
                self.pre_processing.apply(orig_img, lamp_event)

            try:
                res_data = self.frame.process(steps['filled'])
                binary_result = res_data.size > 0
            except Frame.FrameIsEmpty:
                res_data = self.empty
                binary_result = False

            av_bin_result = self.mean_tracker.update(binary_result)
            if av_bin_result:
                self.sl_app_conn.switch_on_lamp()

            if self.save_flag:
                packed_data = self.prepare_array_to_save(res_data, iterator, av_bin_result, lamp_status)
                if self.config['save_csv']:
                    self.save_csv.write(packed_data)
                if self.config['save_img'] or self.config['stream']['enabled']:
                    self.save_img.write(steps, packed_data, iterator, lamp_status)

            self.time_measurements.append(timeit.default_timer() - start_time)

            iterator += 1

            if iterator % self.time_window == 0:
                mean_fps = round(1 / (sum(self.time_measurements) / self.time_window), 1)
                logger.info("FPS for last {} samples: mean - {}".format(self.time_window, mean_fps))
                logger.info("Processed images for all time: {} ".format(iterator))
                self.time_measurements = list()

        if self.config['save_csv']:
            self.save_csv.quit()

        if self.config['stream']['enabled']:
            self.save_img.quit()

        logger.info('Detection finished, {} images processed'.format(iterator))


class Frame(object):
    class Decorators(object):
        @classmethod
        def check_input_on_empty_arr(cls, decorated):
            """
            Executes some detection stage (e.g. filtering) if passed array is not empty, otherwise interrupts iteration
            :param decorated: detection function
            :return: mutated array of parameters
            """
            @wraps(decorated)
            def wrapper(*args, **kwargs):
                return decorated(*args, **kwargs) if args[1].size > 0 else Frame.FrameIsEmpty.interrupt_cycle()
            return wrapper

        @classmethod
        def check_on_conf_flag(cls, decorated):
            """
            Executes detection function if corresponding parameter in config is true (>0), otherwise returns original
            array of parameters
            :param decorated: detection function
            :return: original array of parameters or mutated array of parameters
            """
            @wraps(decorated)
            def wrapper(*args, **kwargs):
                return decorated(*args) if kwargs['dec_flag'] else args[1]
            return wrapper

    class FrameIsEmpty(Exception):
        """
        Used to interrupt the processing at any stage when no more objects are remaining in the parameters array (e.g
        due to preliminary filtering)
        """

        def __init__(self):
            Exception.__init__(self, 'No objects in frame are present')

        @staticmethod
        def interrupt_cycle():
            raise Frame.FrameIsEmpty

    def __init__(self, config):
        self.res = config['resolution']

        self.img_area_px = self.res[0] * self.res[1]
        self.c_ar_thr = config['cont_area_thr']

        self.margin_offset = config['margin']
        self.left_mar, self.right_mar = self.margin_offset, self.res[0] - self.margin_offset
        self.up_mar, self.bot_mar = self.margin_offset, self.res[1] - self.margin_offset

        self.extent_thr = config['extent_thr']
        self.max_dist_thr = config['max_distance']

    @Decorators.check_input_on_empty_arr
    def find_basic_params(self, mask):
        cnts, _ = cv2.findContours(mask, mode=0, method=1)
        c_areas = np.asarray([cv2.contourArea(cnt) for cnt in cnts])
        b_rects = np.asarray([cv2.boundingRect(b_r) for b_r in cnts])

        return np.column_stack((c_areas, b_rects))

    @Decorators.check_input_on_empty_arr
    def calc_second_point(self, temp_param):
        p2_x = temp_param[:, 1] + temp_param[:, 3]
        p2_y = temp_param[:, 2] + temp_param[:, 4]

        return np.column_stack((temp_param, p2_x, p2_y)).astype(np.float32)

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_c_ar(self, basic_params):
        # Filter out small object below threshold
        basic_params = basic_params[basic_params[:, 0] / self.img_area_px > self.c_ar_thr]
        return basic_params

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_margin(self, basic_params):
        margin_filter_mask = ((basic_params[:, 1] > self.left_mar) &  # Built filtering mask
                              (basic_params[:, 5] < self.right_mar) &
                              (basic_params[:, 2] > self.up_mar) &
                              (basic_params[:, 6] < self.bot_mar))

        return basic_params[margin_filter_mask]

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_extent(self, basic_params):
        basic_params = basic_params[basic_params[:, 0] / (basic_params[:, 3] * basic_params[:, 4]) > self.extent_thr]
        return basic_params

    def process(self, mask):
        basic_params = self.find_basic_params(mask)
        basic_params = self.calc_second_point(basic_params)
        # Filtering by object contour area size if filtering by contour area size is enabled
        basic_params = self.filter_c_ar(basic_params, dec_flag=self.c_ar_thr)
        # Filtering by intersection with a frame border if filtering is enabled
        basic_params = self.filter_margin(basic_params, dec_flag=self.margin_offset)
        basic_params = self.filter_extent(basic_params, dec_flag=self.extent_thr)

        return basic_params


class MeanResultTracker(object):
    def __init__(self, q_len, true_events):
        self.obj_q = deque(maxlen=q_len)
        self.true_events = true_events

    def update(self, det_result):
        self.obj_q.appendleft(det_result)

        return self.obj_q.count(True) > self.true_events
