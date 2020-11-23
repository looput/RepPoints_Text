import tensorflow as tf
import os.path as osp
from tensorflow.python.framework import ops
import tensorflow.keras as keras
import math

filename = osp.join(osp.dirname(__file__), 'deformable_conv2d.so')
deformable_conv2d_module = tf.load_op_library(filename)
deformable_conv2d_op = deformable_conv2d_module.deformable_conv2d
deformable_conv2d_grad_op = deformable_conv2d_module.deformable_conv2d_back_prop


@ops.RegisterGradient("DeformableConv2D")
def _deformable_conv2d_back_prop(op, grad):
    """The gradients for `deform_conv`.
        Args:
        op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `roi_pool` op.
        Returns:
        Gradients with respect to the input of `deform_conv`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]
    mask = op.inputs[3]
    '''
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: {'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    strides = op.get_attr('strides')
    dilations = op.get_attr('dilations')
    data_format = op.get_attr('data_format')
    im2col_step = op.get_attr('im2col_step')
    no_bias = op.get_attr('no_bias')
    pads = op.get_attr('padding')
    num_groups = op.get_attr('num_groups')
    deformable_groups = op.get_attr('deformable_groups')
    '''
    REGISTER_OP("DeformableConv2DBackProp")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Input("mask: T")
        .Input("out_grad: T")
        .Output("x_grad: T")
        .Output("filter_grad: T")
        .Output("offset_grad: T")
        .Output("mask_grad: T")
        .Attr("T: {float, double}")
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: { 'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    # compute gradient
    data_grad = deformable_conv2d_grad_op(data, filter,
                                          offset, mask,
                                          grad, strides=strides,
                                          num_groups=num_groups,
                                          deformable_groups=deformable_groups,
                                          im2col_step=im2col_step,
                                          no_bias=no_bias,
                                          padding=pads,
                                          data_format=data_format,
                                          dilations=dilations)
    # data_grad = deformable_conv2d_grad_op(data, filter, offset, mask, grad, strides, num_groups, deformable_groups, im2col_step, no_bias, pads, data_format, dilations)
    return data_grad  # List of 4 Tensor, since we have 4 input


class DeformableConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), num_groups=1, deformable_groups=1, strides=(1, 1, 1, 1), im2col=1,
                 use_bias=False, padding="VALID", data_format='NCHW', dilations=(1, 1, 1, 1)):
        super(DeformableConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.strides = strides
        self.im2col = im2col
        self.use_bias = use_bias
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def build(self, input_shape):
        channel = int(input_shape[1])
        kernel_h,kernel_w = self.kernel_size
        height,width = input_shape[2:]
        
        # self.mask = tf.constant(1., shape=[1, kernel_h * kernel_w, height, width])
        self.filter = tf.Variable(initial_value=tf.random.normal(shape=[self.filters, channel, self.kernel_size[0], self.kernel_size[1]]))
        self.built = True

    def call(self, x,offset, **kwargs):
        """
        Build static Graph
        :param x: [B,Channel, Height, Width]
        :param offset: [B,n*2, Height, Width]
        :return:
        """
        # 这里mask 并不能直接，因为形状是跟输入有关的
        mask = tf.math.reduce_mean(offset,axis=1,keepdims=True)*0+1.
        mask = tf.tile(mask,(1,self.kernel_size[0]*self.kernel_size[0],1,1))
        result = deformable_conv2d_module.deformable_conv2d(
            input=x,
            filter=self.filter,
            offset=offset,
            mask=mask,
            strides=self.strides,
            num_groups=self.num_groups,
            deformable_groups=self.deformable_groups,
            im2col_step=self.im2col,
            no_bias=(not self.use_bias),
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations)
        return result

if __name__ == '__main__':
    tf.enable_eager_execution()
    height = 10
    width = 5
    kernel_h = 3
    kernel_w = 3
    padding = "SAME"
    # input = tf.random.uniform(shape=[1, 3, height, width], maxval=10)
    # with tf.GradientTape(persistent=True) as tape:
    #     # input = tf.ones(shape=[1, 1, height, width])
    #     input = tf.random.uniform(shape=[1, 1, height, width], maxval=10)
    #     tape.watch(input)
    #     input_b = tf.pad(input, [[0, 0], [0, 0], [1, 1], [1, 1]])
    #     filter = tf.Variable(tf.random.uniform(shape=[kernel_h, kernel_w, 1, 1], maxval=10))
    #     filter_deform = tf.transpose(filter, [3, 2, 0, 1])
    #     offset = tf.constant([0. for i in range(kernel_h * kernel_w * 2 * height * width)],
    #                          shape=[1, kernel_h * kernel_w * 2, height, width])
    #     mask = tf.constant([1. for i in range(kernel_h * kernel_w * height * width)],
    #                        shape=[1, kernel_h * kernel_w, height, width])
    #     result = deformable_conv2d_module.deformable_conv2d(
    #         input=input,
    #         filter=filter_deform,
    #         offset=offset,
    #         mask=mask,
    #         strides=[1, 1, 1, 1],
    #         num_groups=1,
    #         deformable_groups=1,
    #         im2col_step=1,
    #         no_bias=True,
    #         padding=padding,
    #         data_format='NCHW',
    #         dilations=[1, 1, 1, 1])
    #     # conv2d = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding, data_format='NCHW')
    #     grad1 = tape.gradient(result, input)
        # grad2 = tape.gradient(conv2d, input)
        # print(input)
        # print(grad1)
        # print(grad2)
    with tf.GradientTape() as tape:
        input_data = tf.random.uniform(shape=[64, 100, 100, 3])
        tape.watch(input_data)
        results = DeformableConv2D(128)(input_data)
        print(tf.shape(results))
