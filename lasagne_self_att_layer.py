import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import MergeLayer
from lasagne.nonlinearities import softmax, tanh

class self_att_layer(MergeLayer):

    def __init__(self, incoming, num_units, mask_input=None, W1=init.GlorotUniform(), b1=init.Constant(0.), **kwargs):

        incomings = [incoming]

        self.nu = num_units
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(self_att_layer, self).__init__(incomings, **kwargs)

        self.W1 = self.add_param(W1, (num_units, num_units), name='W1')
#        self.W2 = self.add_param(W2, (400, num_units), name='W2')
        #self.W2 = self.add_param(W, (400, num_units), name='W2')
        self.b1 = self.add_param(b1, (num_units, ), name='b1', regularizable=False)
#        self.b2 = self.add_param(b2, (num_units, ), name='b2', regularizable=False)

    def get_output_shape_for(self, input_shapes):

        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], input_shape[2]

    def get_output_for(self, inputs, **kwargs):

        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        (d1, d2, d3) = input.shape

        # out = T.tensordot(input, self.W, axes=[[2], [0]])
        # b_shuffled = self.b.dimshuffle('x', 'x', 0)
        # out += b_shuffled
        # out = tanh(out)
        # out *= mask.dimshuffle(0, 1, 'x')
        # out = T.batched_dot(out, out.dimshuffle(0, 2, 1))
        q = T.tensordot(input, self.W1, axes=[[2], [0]])
        b1_shuffled = self.b1.dimshuffle('x', 'x', 0)
        q += b1_shuffled
        q = tanh(q)

#        k = T.tensordot(input, self.W2, axes=[[2], [0]])
        # b2_shuffled = self.b2.dimshuffle('x', 'x', 0)
        # k += b2_shuffled
        # k = tanh(k)

        q *= mask.dimshuffle(0, 1, 'x')
#        k *= mask.dimshuffle(0, 1, 'x')
        out = T.batched_dot(q, q.dimshuffle(0, 2, 1))
        #out /= np.sqrt(self.nu)
        #out *= 0.1

        out *= (1 - T.eye(d2, d2))
        
        matrix = softmax(out.reshape((d1 * d2, d2))).reshape((d1, d2, d2))
        matrix *= mask.dimshuffle(0, 1, 'x')
        matrix *= mask.dimshuffle(0, 'x', 1)

        return matrix

class dotlayer(MergeLayer):

    def __init__(self, incoming1, incoming2, mask_input=None, nonlinearity=None, reverse=False, **kwargs):

        incomings = [incoming1, incoming2]
        self.mask_incoming_index = -1
        self.nonlinearity = nonlinearity
        self.reverse = reverse
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 2

        super(dotlayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):

        input_shape = input_shapes[1]
        return input_shape[0], input_shape[1], input_shape[2]

    def get_output_for(self, inputs, **kwargs):

        input1 = inputs[0]
        input2 = inputs[1]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        input1 *= mask.dimshuffle(0, 1, 'x')
        if self.reverse:
            input2 *= mask.dimshuffle(0, 'x', 1)
        else :
            input2 *= mask.dimshuffle(0, 1, 'x')
        out = T.batched_dot(input1, input2)
        out *= mask.dimshuffle(0, 1, 'x')

        if self.nonlinearity is not None:
            (d1, d2, d3) = input1.shape
            out = self.nonlinearity(out.reshape((d1 * d2, d2))).reshape((d1, d2, d2))
            out *= mask.dimshuffle(0, 1, 'x')
            out *= mask.dimshuffle(0, 'x', 1)

        return out

