from styx_msgs.msg import TrafficLight

import cv2
import numpy as np
import os
import rospy
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        self.method = rospy.get_param("/traffic_light_classifier_method")

        if self.method == "cv":
            rospy.loginfo('using CV classifier')
        elif self.method == "cv2":
            rospy.loginfo('using CV classifier v2')
        elif self.method == "nn":
            rospy.loginfo('using NN classifier')
            model = rospy.get_param("/traffic_light_classifier_model")
            
            dir_path = os.path.dirname(os.path.realpath(__file__)) + "/nn"

            self.sess = tf.Session()

            model_dir = dir_path + '/' + model + '/'
            model_file = model_dir + 'ckp_tf_classifier.meta'
            rospy.loginfo('loading model from: %s', model_file)

            saver = tf.train.import_meta_graph(model_file)
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

            self.graph = tf.get_default_graph()
            self.tl_logits = self.graph.get_tensor_by_name("tl_logits/BiasAdd:0")
        else:
            assert False, ("Method not available: %s" % (self.method))


    def get_classification(self, image):
        if self.method == "cv":
            val = self.get_classification_cv(image)
        elif self.method == "cv2":
            val = self.get_classification_cv2(image)
        elif self.method == "nn":
            val = self.get_classification_nn(image)
        else:
            assert False, ("Method not available: %s" % (self.method))

        return val


    def get_classification_cv(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
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


    def get_classification_cv2(self, i):
        lb = np.array([100, 190, 255])
        ub = np.array([190, 255, 255])

        m = cv2.inRange(i, lb, ub)
        i = cv2.bitwise_and(i, i, mask=m)

        lb = np.array([119, 131, 255])
        ub = np.array([153, 255, 255])

        m = cv2.inRange(i, lb, ub)
        i = cv2.bitwise_and(i, i, mask=m)

        mx = 0
        (lx, ly, lz ) = i.shape
        for x in range(0, lx, 10):
            for y in range(0, ly, 10):
                c = np.count_nonzero(i[x:x+30,y:y+30,0])
                if c > mx:
                    mx = c

        if mx > 150:
            return TrafficLight.RED

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