# -*- coding: utf-8 -*-
# File: common.py

import numpy as np
import cv2

from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import ImageAugmentor, ResizeTransform,Transform
from tensorpack.dataflow.imgaug.transform import WarpAffineTransform


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class CustomResize(ImageAugmentor):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(CustomResize, self).__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        size = self.rng.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)

class Random_Resize(ImageAugmentor):
    ''' 图片的尺寸应能整除一些数字
    '''
    def __init__(self, short_edge_length, max_size,ratios=(3/4.,4/3.),divs=32, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(Random_Resize, self).__init__()
        self.divs = divs
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        size = self.rng.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        rt = self.rng.rand()*(self.ratios[1]-self.ratios[0])+self.ratios[0]
        rt = np.sqrt(rt)
        rt = np.sqrt(rt)
        # rt=1
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww/rt + 0.5)
        newh = int(newh*rt + 0.5)
        
        neww = ((neww+self.divs-1)//self.divs)*self.divs
        newh = ((newh+self.divs-1)//self.divs)*self.divs
        return ResizeTransform(h, w, newh, neww, self.interp)

class CusRotation(ImageAugmentor):
    """ Random rotate the image w.r.t a random center
        原始版本会切除图像，避免这种情况进行resize
    """

    def __init__(self, max_deg, center_range=(0, 1),
                 interp=cv2.INTER_LINEAR,
                 border=cv2.BORDER_REPLICATE, step_deg=None, border_value=0):
        assert step_deg is None or (max_deg == 180 and max_deg % step_deg == 0)
        super(CusRotation, self).__init__()
        self._init(locals())

    def get_transform(self, img):
        center = img.shape[1::-1] * self._rand_range(
            self.center_range[0], self.center_range[1], (2,))
        deg = self._rand_range(-self.max_deg, self.max_deg)
        if self.step_deg:
            deg = deg // self.step_deg * self.step_deg
        # print(deg)
        matrix = cv2.getRotationMatrix2D(tuple(center - 0.5), deg, 1)

        width, height = img.shape[1::-1]
        cos = np.abs(matrix[0,0])
        sin = np.abs(matrix[0,1])
        new_W = int((height * sin) + (width * cos))
        new_H = int((height * cos) + (width * sin))

        matrix[0,2] += (new_W/2) - width/2
        matrix[1,2] += ((new_H/2)) - height/2

        return WarpAffineTransform(
            matrix, (new_W,new_H), interp=self.interp,
            borderMode=self.border, borderValue=self.border_value)

class Resize(ImageAugmentor):
    ''' 图片的尺寸应能整除一些数字
    '''
    def __init__(self, min_length, max_length,keep_ratio=False,divs=32, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(Resize, self).__init__()
        self.divs = divs
        self.min_length = min_length
        self.max_length = max_length
        self.keep_ratio = keep_ratio
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        if not self.keep_ratio:
            if h < w:
                newh, neww = self.min_length, self.max_length
            else:
                newh, neww = self.max_length, self.min_length
        else:
            sacle = self.max_length/max(h,w)
            # sacle = self.min_length/min(h,w)
            newh,neww = h*sacle,w*sacle
        neww = int(neww)
        newh = int(newh)
        
        neww = ((neww+self.divs-1)//self.divs)*self.divs
        newh = ((newh+self.divs-1)//self.divs)*self.divs
        return ResizeTransform(h, w, newh, neww, self.interp)

class OrderTransform(Transform):
    """
        调整polygon点的顺序
    """
    def __init__(self):
        self._init(locals())

    def apply_image(self, img):
        return img

    def apply_coords(self, coords):
        # 总共分两步：
        # 确保点的顺序为顺时针
        # 确定起点
        return coords

class Sort_Order(ImageAugmentor):
    """
        确保polygon使用合适的顺序
    """
    def __init__(self, horiz=False, vert=False, prob=0.5):
        """
        Args:
            horiz (bool): use horizontal flip.
            vert (bool): use vertical flip.
            prob (float): probability of flip.
        """
        super(Sort_Order, self).__init__()
        self._init(locals())

    def get_transform(self, img):
        return NoOpTransform()

def box_to_point4(boxes):
    """
    Convert boxes to its corner points.

    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point4_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


try:
    import pycocotools.mask as cocomask

    # Much faster than utils/np_box_ops
    def np_iou(A, B):
        def to_xywh(box):
            box = box.copy()
            box[:, 2] -= box[:, 0]
            box[:, 3] -= box[:, 1]
            return box

        ret = cocomask.iou(
            to_xywh(A), to_xywh(B),
            np.zeros((len(B),), dtype=np.bool))
        # can accelerate even more, if using float32
        return ret.astype('float32')

except ImportError:
    from utils.np_box_ops import iou as np_iou  # noqa
