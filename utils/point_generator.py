import tensorflow as tf

class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, feat_h, feat_w, stride=16):
        shift_x = tf.range(0., feat_w) * stride
        shift_y = tf.range(0., feat_h) * stride
        shift_xx, shift_yy = tf.meshgrid(shift_x, shift_y)
        # stride = tf.constant((shift_xx.shape[0], ), stride)
        stride = tf.cast(tf.fill(tf.shape(shift_xx), stride),tf.float32)
        all_points = tf.stack([shift_xx, shift_yy,stride], axis=-1)
        all_points = tf.reshape(all_points,[-1,3])
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = tf.zeros(feat_w, dtype=tf.uint8)
        valid_y = tf.zeros(feat_h, dtype=tf.uint8)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = tf.meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid
