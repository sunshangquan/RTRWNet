

def compute_occulusion_using_pos(fore, back, mask_threshold=1.0):
    h, w, d = fore.shape
    occulusion = numpy.zeros((h, w), dtype="uint8")
    warp_lib.fast_compute_occulusion(
        occulusion.ctypes.data_as(ctypes.c_char_p), 
        fore.ctypes.data_as(ctypes.c_char_p), 
        back.ctypes.data_as(ctypes.c_char_p), 
        h, w, d,
        # 浮点数最好显示转换
        ctypes.c_float(mask_threshold)
    )
    return occulusion


occulusion_pos_threshold = 1.5
forward_occulusion_using_pos  = compute_occulusion_using_pos(forward_flow,  backward_flow, mask_threshold=occulusion_pos_threshold)
backward_occulusion_using_pos = compute_occulusion_using_pos(backward_flow, forward_flow,  mask_threshold=occulusion_pos_threshold)
cv_write(add_to_save("forward_occulusion_using_pos.png"),  forward_occulusion_using_pos)
cv_write(add_to_save("backward_occulusion_using_pos.png"), backward_occulusion_using_pos)
cv_show(numpy.concatenate([forward_occulusion_using_pos, backward_occulusion_using_pos], axis=0))