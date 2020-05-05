#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None

    def load_model(self, model, device, cpu_extension=None, plugin=None):
        
        # Obtain model files path:
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        ### TODO: Load the model ### Set a self.plugin variable with IECore in case there is no plugin passed.
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_extension(cpu_extension, "CPU")

        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)

        log.info("Loading IR to the plugin...")

        # If applicable, add a CPU extension to self.plugin
        if "CPU" in device:
            ### TODO: Check for supported layers ###
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            ### TODO: Add any necessary extensions ###
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(device,', '.join(not_supported_layers)))
                log.error("Please try to specify another cpu extension library path (via -l or --cpu_extension command line parameters)"
                          " that support required model layers or try, in last case, with other model")
                sys.exit(1)

        # Load the model to the network:
        self.net_plugin = self.plugin.load_network(network=self.net, device_name=device)

        # Obtain other relevant information:
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.plugin


    def get_input_shape(self, faster_rnn=False):
        ### TODO: Return the shape of the input layer ###
        if not faster_rnn:
            return self.net.inputs[self.input_blob].shape
        else:
            return self.net.inputs['image_tensor'].shape


    def exec_net(self, frame, request_id=0, faster_rnn=False):
        ### TODO: Start an asynchronous request ###
        if not faster_rnn:
            self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={self.input_blob: frame})
        else:
            self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, inputs={'image_tensor': frame,'image_info': frame.shape[1:]})

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin


    def wait(self, request_id=0):
        ### TODO: Wait for the request to be complete. ###
        status = self.net_plugin.requests[request_id].wait(-1)

        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self, request_id=0):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.net_plugin.requests[request_id].outputs[self.out_blob]
