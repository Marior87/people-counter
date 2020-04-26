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
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        
        ### TODO: Load the model ### Set a self.plugin variable with IECore
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
        
        # If applicable, add a CPU extension to self.plugin # Why adding this here if only necessary in case there unsupported layers?
        #if cpu_extension and "CPU" in device:
            #self.plugin.add_extension(cpu_extension, device)
        
        # Load the Intermediate Representation model
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        
        # Check supported layers:
        supported_layers = self.plugin.query_network(ir_net)
        supported_layers = set(supported_layers.keys()) # Only to get supported layers of given Network.
        
        # Check required network layers:
        network_layers = set(self.network.layers) # To have all required layers.
        
        # Check if there aren't unsupported layers:
        differences = supported_layers.symmetric_difference(network_layers) # Comparing both sets.
        
        if len(differences)>0:
            print('Following layers are not supported: ',*differences, '. Will proceed to add CPU Extension.')
            
            ### TODO: Add any necessary extensions ###
            self.plugin.add_extension(CPU_EXTENSION,'CPU')
        
            # Checking again that there are not any more unsupported layers:
            supported_layers = self.plugin.query_network(ir_net,device_name='CPU')
            supported_layers = set(supported_layers.keys())
            differences = supported_layers.symmetric_difference(network_layers)
            print('Extension added, unsupported layer(s) now: '+str(len(differences)))
        else:
            print('All layers are supported, no extensions required.')
          
        # Get other relevant information:
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.plugin.load_network(self.network, device)

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        self.exec_network.start_async(request_id=0, inputs={self.input_blob: image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
