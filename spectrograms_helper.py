
import numpy as np 
import spec_ops as spec_ops
import phase_operation as phase_op

def _linear_to_mel_matrix():
    """Get the mel transformation matrix."""
    _sample_rate=16000
    _mel_downscale=1
    num_freq_bins = 2048 // 2
    lower_edge_hertz = 0.0
    upper_edge_hertz = 16000 / 2.0
    num_mel_bins = num_freq_bins // _mel_downscale
    return spec_ops.linear_to_mel_weight_matrix(
        num_mel_bins, num_freq_bins, _sample_rate, lower_edge_hertz,
        upper_edge_hertz)

def _mel_to_linear_matrix():
    """Get the inverse mel transformation matrix."""
    m = _linear_to_mel_matrix()
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))



def melspecgrams_to_specgrams(logmelmag2, mel_p):
    """Converts melspecgrams to specgrams.
    Args:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    Returns:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    """
    logmelmag2 = logmelmag2.T
    mel_p = mel_p.T
    logmelmag2 = np.array([logmelmag2])
    mel_p = np.array([mel_p])
  
    
    mel2l = _mel_to_linear_matrix()  
    mag2 = np.tensordot(np.exp(logmelmag2), mel2l, 1)
    logmag = 0.5 * np.log(mag2+1e-6)
    mel_phase_angle = np.cumsum(mel_p * np.pi, axis=1)
    phase_angle = np.tensordot(mel_phase_angle, mel2l, 1)
    p = phase_op.instantaneous_frequency(phase_angle,time_axis=1)
    return logmag[0].T, p[0].T


def specgrams_to_melspecgrams(magnitude, IF):
    """Converts specgrams to melspecgrams.
    Args:
      specgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time].
    Returns:
      melspecgrams: Tensor of log magnitudes and instantaneous frequencies,
        shape [freq, time], mel scaling of frequencies.
    """
    logmag = magnitude.T
    p = IF.T
    mag2 = np.exp(2.0 * logmag)
    mag2 = np.array([mag2])
    phase_angle = np.cumsum(p * np.pi, axis=1)
    phase_angle = np.array([phase_angle])

    l2mel = _linear_to_mel_matrix()
    logmelmag2 = np.log(np.tensordot(mag2,l2mel,axes=1) + 1e-6)
    mel_phase_angle = np.tensordot(phase_angle, l2mel, axes=1)
    mel_p = phase_op.instantaneous_frequency(mel_phase_angle,time_axis=1)
    return logmelmag2[0].T, mel_p[0].T