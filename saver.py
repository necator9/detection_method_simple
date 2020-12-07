# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Saving detection output as csv or images.

import os
import numpy as np
import cv2
import logging
import ffmpeg
import socket
import time

logger = logging.getLogger('detect.ext')


class SaveCSV(object):
    """
    Save detection information to csv file
    """
    def __init__(self, out_dir):
        # img_num, o_num, c_a, x, y, w, h, p2x, p2y, av_bin_res, lamp_status
        self.fd = open(os.path.join(out_dir, 'detected_objects.csv'), 'w')
        self.fmt = '%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d'
        # self.fd.write("img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,ca,x,y,w,h,p2x,p2y,o_prob,o_class,av_bin,lamp\n")
        self.fd.write("img,o_num,c_a,x,y,w,h,p2x,p2y,av_bin,lamp\n")

    def write(self, data):
        if data.size > 0:
            np.savetxt(self.fd, data, fmt=self.fmt)

    def quit(self):
        self.fd.close()


class SaveImg(object):
    """
    Save image to file or/and stream
    """
    def __init__(self, config):
        self.config = config

        self.o_color = (0, 255, 0)
        self.padding = 30

        width, height = [(d + (self.padding * 2)) * 2 for d in config['resolution']]

        if self.config['stream']['enabled']:
            stream_server = '{}/{}_{}'.format(config['stream']['server'], socket.gethostname(), config['device'])
            self.process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
                            .output(stream_server, vcodec='mpeg4', format='rtsp', video_bitrate='1000', framerate='5')
                            .overwrite_output()
                            .run_async(pipe_stdin=True))

    def write(self, steps, data, iterator, lamp_status):
        out_img = self.prepare_multiple_img(steps, data, lamp_status)
        if self.config['stream']['enabled']:
            self.process.stdin.write(out_img.astype(np.uint8).tobytes())
        if self.config['save_img']:
            cv2.imwrite(os.path.join(self.config['out_dir'], '{}.jpeg'.format(iterator)), out_img)

    def prepare_multiple_img(self, steps, data, lamp_status):
        """
        Build an output image containing multiple detection stages
        Parameters
        ----------
        steps: dictionary containing detection stages
        data: detection parameters
        lamp_status: status of the lamp on current frame

        Returns
        -------
        out_img: processed and stacked detection stages
        """
        for key, img in steps.items():
            steps[key] = self.prepare_single_img(img, key)

        # Put current time
        current_time = time.strftime("%H:%M:%S  %d.%m.%y", time.localtime())
        cv2.putText(steps['filled'], '{}'.format(current_time), (self.padding, steps['filled'].shape[0] - self.padding),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        self.draw_rects(steps['resized_orig'], data)

        h_stack1 = np.hstack((steps['mask'], steps['filtered']))
        h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))
        out_img = np.vstack((h_stack1, h_stack2))

        if data.size > 0 and data[0][9]:  # Draw detection result on current frame
            cv2.rectangle(out_img, (0, 0), (out_img.shape[1], out_img.shape[0]), (0, 0, 255), 10)
        if lamp_status:  # Draw lamp status
            cv2.rectangle(out_img, (0, 0), (out_img.shape[1], out_img.shape[0]), (0, 255, 255), 3)

        return out_img

    def prepare_single_img(self, img, name):
        img = self.add_padding(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img, name, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def draw_rects(self, img, data):
        """
        Draw detection information on frame
        Parameters
        ----------
        img: image on which drawing is performed
        data: detection information for current frame
        """
        # data = img_num, o_num, c_a, x, y, w, h, p2x, p2y, av_bin_res, lamp_status
        data_frame = data[:, [2, 3, 4, 7, 8]]  # data_frame = c_a, x, y, p2x, p2y
        data_frame[:, [1, 2, 3, 4]] += self.padding
        for ca, x, y, p2x, p2y in data_frame.tolist():
            # if o_class == 0:
            #     continue
            color = self.o_color
            p1x, p2x, p1y, p2y = int(x), int(p2x), int(y), int(p2y)
            cv2.rectangle(img, (p1x, p1y), (p2x, p2y), color, 1)
            cv2.putText(img, '{0}'.format(ca), (p1x, p1y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    def add_padding(self, image):
        """
        Add some black padding around the frame
        Parameters
        ----------
        image: image to process

        Returns
        -------
        stack2: padded image (input.shape != output.shape)
        """
        h_pad = np.zeros((self.padding, self.config['resolution'][0]), np.uint8)
        stack1 = np.vstack((h_pad, image, h_pad))

        v_pad = np.zeros((stack1.shape[0], self.padding), np.uint8)
        stack2 = np.hstack((v_pad, stack1, v_pad))

        return stack2

    def quit(self):
        self.process.stdin.close()
        self.process.wait()
