import pandas as pd
import numpy as np

def compute_composite_snr_db(snr_trad_db, distance_m, rel_speed_ms, 
                             d_ref=1.0, d_min=0.1, 
                             n=2.0,            # path-loss exponent 
                             v_ref=10.0,       # reference speed (m/s) 
                             w_d=1.0, w_v=1.0, # weights for distance and speed penalties 
                             clip_min_db=-30.0, clip_max_db=40.0): 
    """ 
    Returns composite SNR in dB. 
    snr_trad_db: conventional SNR in dB (float or numpy scalar) 
    distance_m: distance in meters (float) 
    rel_speed_ms: relative speed in m/s (can be negative; function uses abs) 
    Other params are tunable. 
    """ 
    # sanitize inputs 
    d = max(distance_m, d_min) 
    v = abs(rel_speed_ms) 

    # distance penalty: 10 * n * log10(d / d_ref) scaled by w_d 
    pen_d_db = w_d * 10.0 * n * np.log10(d / d_ref) 

    # speed penalty: soft-log penalty, scaled by w_v 
    pen_v_db = w_v * 10.0 * np.log10(1.0 + (v / v_ref)) 

    composite = float(snr_trad_db) - float(pen_d_db) - float(pen_v_db) 

    # clip to avoid extreme values 
    composite = max(min(composite, clip_max_db), clip_min_db) 
    return composite 