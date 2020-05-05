"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
import utils

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    if 'faster' in args.model:
        faster_rnn = True
    else:
        faster_rnn = False

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, device=args.device, cpu_extension=args.cpu_extension)

    # We need model required input dimensions:
    required_input_shape = infer_network.get_input_shape(faster_rnn=faster_rnn)
    required_input_width = required_input_shape[2]
    required_input_height = required_input_shape[3]

    ### TODO: Handle the input stream ###
    if args.input != 'CAM':
        try:
            # It seems that OpenCV can use VideoCapture to treat videos and images:
            input_stream = cv2.VideoCapture(args.input)
            length = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            webcamera = False

            # Check if input is an image or video file:
            if length > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print('Not supported image or video file format. Please pass a supported one.')
            exit()

    else:
        input_stream = cv2.VideoCapture(0)
        single_image_mode = False
        webcamera = True

    # We need fps for time related calculations:
    fps = input_stream.get(cv2.CAP_PROP_FPS)

    # We also need input stream width and height:
    stream_width = int(input_stream.get(3))
    stream_height = int(input_stream.get(4))

    not_in_frame = 0 # Counter for Faster RNN:
    frames_for_quit = 10 # Number of consecutive frames we wait until we consider a person is completly out of frame.

    if not single_image_mode:
        ### TODO: Loop until stream is over ###
        # These are tuning values and others required for the counter logic:
        
        ## Tuning, could be asked as possible arguments:
        LOWER_HALF = 0.7 # Fraction of total height a centroid is considered to be in the "lower half"
        RIGHT_HALF = 0.8 # Fraction of total width a centroid is considered to be in the "right half". With 0.87 works but it is too extreme.
        DETECTION_FRAMES = 1 # If current count_frame is divisible by this number, detection model is run.

        count_frame = 0 # Frame counter.
        status_lower_half = False # Status of the lower half.
        status_upper_half = False # Status of the upper half.
        id = 0 # Identifier for people.
        current_person = [] # For storing current person in frame.
        current_time = [0] # For storing last recorded time.

        # Params to send to MQTT Server:
        total_counted = 0 # People counter.
        people_in_frame = 0 # People in frame status.

        while(input_stream.isOpened()):
        
            ### TODO: Read from the video capture ###
            # Read the next frame:
            flag, frame = input_stream.read()

            # Quit if there is no more stream:
            if not flag:
                break

            # Quit if 'q' is pressed:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Execute detection model if required in this frame:
            if count_frame % DETECTION_FRAMES == 0:

                ### TODO: Pre-process the image as needed ###
                preprocessed_frame = utils.handle_image(frame, width=required_input_width, height=required_input_height)

                ### TODO: Start asynchronous inference for specified request ###
                infer_network.exec_net(preprocessed_frame, faster_rnn=faster_rnn)

                ### TODO: Wait for the result ###
                status = infer_network.wait()
                if status == 0: # Wait until we have results.
                    prev_results = infer_network.get_output() # Get outputs.

                    ### TODO: Get the results of the inference request ###
                    results_bb = []
                    for p_r in prev_results[0,0]: # Iterate over outputs.
                        if p_r[2] >= args.prob_threshold and p_r[1]==1.0: # Filter relevant outputs. p_r[1]==1: check only for people.
                            results_bb.append(p_r[3:]) # Save those relevant results.

                    ### TODO: Extract any desired stats from the results ###
                    if not faster_rnn:# Faster RNN has better detection capabilities, not necessary to porcess the same way.
                        if len(results_bb) > 0:
                            for detection in results_bb: # Iterate through each detection:
                                centroid = utils.calculate_centroid(detection)
                                frame = utils.draw_bounding_box(frame, detection)

                                if centroid[1] > LOWER_HALF and status_lower_half == False and status_upper_half == False: # Meaning there is a new detection in the lower border.
                                    status_lower_half = True
                                    person = utils.Person(id=id, frame_init = count_frame)
                                    current_person.append(person)
                                    total_counted = total_counted + 1
                                    id = id + 1
                                elif status_lower_half:
                                    status_lower_half = False
                                    status_upper_half = True

                                # To check that there is a detection in one of the halves:
                                people_in_frame = status_upper_half + status_lower_half

                                if centroid[0] > RIGHT_HALF and status_upper_half == True:
                                    status_lower_half = False
                                    status_upper_half = False
                                    people_in_frame = 0
                                    current_time[0] = (count_frame - current_person[0].frame_init)/fps
                                    
                                    current_person = []
                                    client.publish("person/duration", json.dumps({"duration": current_time[0]}))
                    else: # Using Faster RNN Model:
                        if len(results_bb) == 0:
                            not_in_frame = not_in_frame + 1
                            if not_in_frame >= frames_for_quit and current_person:
                                not_in_frame = 0
                                people_in_frame = 0
                                if current_person:
                                    # Substracting 'frames_for_quit' because we stopped detecting this person those "frames ago"
                                    current_time[0] = (count_frame - current_person[0].frame_init - frames_for_quit)/fps
                                current_person = []
                                client.publish("person/duration", json.dumps({"duration": current_time[0]}))
                        else:
                            people_in_frame = 1
                            not_in_frame = 0
                            for detection in results_bb: # Iterate through each detection:
                                frame = utils.draw_bounding_box(frame, detection)

                            if not current_person:# Meaning that there is no recorded person.
                                person = utils.Person(id=id, frame_init = count_frame)
                                current_person.append(person)
                                total_counted = total_counted + 1
                                id = id + 1
                            

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": people_in_frame, "total": total_counted}))

            # Additional feature: Change timer color when a person is more than 15 secs on screen.
            if people_in_frame:
                current_time[0] = (count_frame - current_person[0].frame_init)/fps
                if current_time[0] > 15:
                    font_color = (0,0,255)
                else:
                    font_color = (0,0,0)
                frame = utils.draw_text(frame,"Current person: "+str(current_time[0])+" secs",font_color=font_color)
            else:
                if current_time[0] > 15:
                    font_color = (0,0,255)
                else:
                    font_color = (0,0,0)
                frame = utils.draw_text(frame,"Last person: "+str(current_time[0])+" secs",font_color=font_color)

            
            ### TODO: Send the frame to the FFMPEG server ###
            if not webcamera:
                sys.stdout.buffer.write(frame)
            else:
                cv2.imshow('Resultado', frame)

            count_frame = count_frame + 1

        # Release resources:
        input_stream.release()
    else:
        flag, frame = input_stream.read()
        preprocessed_frame = utils.handle_image(frame, width=required_input_width, height=required_input_height)

        infer_network.exec_net(preprocessed_frame)

        status = infer_network.wait()
        if status == 0: # Wait until we have results.
            prev_results = infer_network.get_output() # Get outputs.

            results_bb = []
            for p_r in prev_results[0,0]: # Iterate over outputs.
                if p_r[2] >= args.prob_threshold and p_r[1]==1.0: # Filter relevant outputs. p_r[1]==1: check only for people.
                    results_bb.append(p_r[3:]) # Save those relevant results.

            if len(results_bb) > 0:
                for detection in results_bb: # Iterate through each detection:
                    frame = utils.draw_bounding_box(frame, detection)

            frame = utils.draw_text(frame,"People in Frame: "+str(len(results_bb)),coordinates=(0.05, 0.05))
            cv2.imwrite('result_single_image.png', frame)

    cv2.destroyAllWindows()

    # Disconnect from MQTT:
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
