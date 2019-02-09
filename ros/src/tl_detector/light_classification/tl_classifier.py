from styx_msgs.msg import TrafficLight

import cv2
import numpy as np
import os
import rospy
import tensorflow as tf

class TLClassifier(object):
    def __init__(self, model = 'nn'):
        self.model = rospy.get_param("/traffic_light_classifier_model")
        if self.model == "cv":
            print('using CV classifier')
        else:
            model = 'lr0.0001_95acc'
            print('using NN classifier:', model)

            dir_path = os.path.dirname(os.path.realpath(__file__)) + "nn/"

            self.sess = tf.Session()
            print(dir_path + '/' + model + '/ckp_tf_classifier.meta')
            saver = tf.train.import_meta_graph(dir_path + '/' + model + '/ckp_tf_classifier.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(dir_path + '/' + model + '/'))

            self.graph = tf.get_default_graph()
            self.tl_logits = self.graph.get_tensor_by_name("tl_logits/BiasAdd:0")


    def get_classification(self, image):
        if self.model == "cv":
            val = self.get_classification_cv(image)
        else:
            val = self.get_classification_nn(image)

        return val


    def get_classification_cv(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,150,100])
        upper_red = np.array([20,255,255])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_red, upper_red)
        count = np.count_nonzero(mask)
        if count > 50:
            # print ('predicting', "red", count)
            return TrafficLight.RED
        # print ('predicting', "go")

        return TrafficLight.UNKNOWN


    def get_classification_nn(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = image[200:800, 284:1084, :]

        image_input = self.graph.get_tensor_by_name("image_input:0")
        is_training = self.graph.get_tensor_by_name("is_training:0")
        feed_dict = {image_input: [image, ], is_training: False}
        logits = self.sess.run(self.tl_logits, feed_dict)

        predict = np.argmax(logits[0])
        if predict != TrafficLight.RED:
            predict = TrafficLight.UNKNOWN

        #print ('predicting', predict)
        return predict