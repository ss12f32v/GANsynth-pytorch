import librosa
import numpy as np
from intervaltree import Interval,IntervalTree


def diff(x, axis):
    """Take the finite difference of a tensor along an axis.
    Args:
    x: Input tensor of any dimension.
    axis: Axis on which to take the finite difference.
    Returns:
    d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
    ValueError: Axis out of range for tensor.
    """
    shape = x.shape

    begin_back = [0 for unused_s in range(len(shape))]
#     print("begin_back",begin_back)
    begin_front = [0 for unused_s in range(len(shape))]

    begin_front[axis] = 1
#     print("begin_front",begin_front)

    size = list(shape)
    size[axis] -= 1
#     print("size",size)
    slice_front = x[begin_front[0]:begin_front[0]+size[0], begin_front[1]:begin_front[1]+size[1]]
    slice_back = x[begin_back[0]:begin_back[0]+size[0], begin_back[1]:begin_back[1]+size[1]]

#     slice_front = tf.slice(x, begin_front, size)
#     slice_back = tf.slice(x, begin_back, size)
#     print("slice_front",slice_front)
#     print(slice_front.shape)
#     print("slice_back",slice_back)

    d = slice_front - slice_back
    return d


def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
    p: Phase tensor.
    discont: Float, size of the cyclic discontinuity.
    axis: Axis of which to unwrap.
    Returns:
    unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
#     print("dd",dd)
    ddmod = np.mod(dd+np.pi,2.0*np.pi)-np.pi  # ddmod = tf.mod(dd + np.pi, 2.0 * np.pi) - np.pi
#     print("ddmod",ddmod)

    idx = np.logical_and(np.equal(ddmod, -np.pi),np.greater(dd,0)) # idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
#     print("idx",idx)
    ddmod = np.where(idx, np.ones_like(ddmod) *np.pi, ddmod) # ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
#     print("ddmod",ddmod)
    ph_correct = ddmod - dd
#     print("ph_corrct",ph_correct)
    
    idx = np.less(np.abs(dd), discont) # idx = tf.less(tf.abs(dd), discont)
    
    ddmod = np.where(idx, np.zeros_like(ddmod), dd) # ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=axis) # ph_cumsum = tf.cumsum(ph_correct, axis=axis)
#     print("idx",idx)
#     print("ddmod",ddmod)
#     print("ph_cumsum",ph_cumsum)
    
    
    shape = np.array(p.shape) # shape = p.get_shape().as_list()

    shape[axis] = 1
    ph_cumsum = np.concatenate([np.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis) 
    #ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
#     print("unwrapped",unwrapped)
    return unwrapped


def instantaneous_frequency(phase_angle, time_axis):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Unwrap and take the finite difference of the phase. Pad with initial phase to
    keep the tensor the same size.
    Args:
    phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
    time_axis: Axis over which to unwrap and take finite difference.
    Returns:
    dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
#     print("phase_unwrapped",phase_unwrapped.shape)
    
    dphase = diff(phase_unwrapped, axis=time_axis)
#     print("dphase",dphase.shape)
    
    # Add an initial phase to dphase
    size = np.array(phase_unwrapped.shape)
#     size = phase_unwrapped.get_shape().as_list()

    size[time_axis] = 1
#     print("size",size)
    begin = [0 for unused_s in size]
#     phase_slice = tf.slice(phase_unwrapped, begin, size)
#     print("begin",begin)
    phase_slice = phase_unwrapped[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1]]
#     print("phase_slice",phase_slice.shape)
    dphase = np.concatenate([phase_slice, dphase], axis=time_axis) / np.pi

#     dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase


def polar2rect(mag, phase_angle):
    """Convert polar-form complex number to its rectangular form."""
#     mag = np.complex(mag)
    temp_mag = np.zeros(mag.shape,dtype=np.complex_)
    temp_phase = np.zeros(mag.shape,dtype=np.complex_)

    for i, time in enumerate(mag):
        for j, time_id in enumerate(time):
#             print(mag[i,j])
            temp_mag[i,j] = np.complex(mag[i,j])
#             print(temp_mag[i,j])
    
    for i, time in enumerate(phase_angle):
        for j, time_id in enumerate(time):
            temp_phase[i,j] = np.complex(np.cos(phase_angle[i,j]), np.sin(phase_angle[i,j]))
#             print(temp_mag[i,j])
    
#     phase = np.complex(np.cos(phase_angle), np.sin(phase_angle))
   
    return temp_mag * temp_phase
