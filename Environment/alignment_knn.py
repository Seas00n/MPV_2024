from feature_extra_new import *


def align_fea(pcd_new: pcd_opreator_system,
              pcd_pre: pcd_opreator_system,
              _print_=False):
    pcd_to_align_new = np.zeros((0, 2))
    pcd_to_align_pre = np.zeros((0, 2))
    fea_component_new = [0, 0, 0]
    fea_component_pre = [0, 0, 0]
    pcd_align_flag = 1
    if pcd_new.fea_extra_over and pcd_pre.fea_extra_over:
        if (pcd_new.corner_situation == 5 and pcd_pre.corner_situation == 1) or (
                pcd_new.corner_situation == 6 and pcd_pre.corner_situation == 1) or (
                pcd_new.corner_situation == 4 and pcd_pre.corner_situation == 1):
            if pcd_new.is_fea_C_gotten and pcd_pre.is_fea_A_gotten:
                fea_len = min(np.shape(pcd_new.fea_C)[0], np.shape(pcd_pre.fea_A)[0])
                pcd_to_align_new = np.vstack([pcd_to_align_new, pcd_new.fea_C[0:fea_len, :]])
                pcd_to_align_pre = np.vstack([pcd_to_align_pre, pcd_pre.fea_A[0:fea_len, :]])
                if _print_:
                    print("fea_C of new pcd and fea_A of pre pcd:{}".format(fea_len))
                fea_component_new = [0, 0, fea_len]
                fea_component_pre = [fea_len, 0, 0]
                pcd_align_flag = 0
                return pcd_to_align_new, pcd_to_align_pre, pcd_align_flag
        elif (pcd_new.corner_situation == 1 and pcd_pre.corner_situation == 5) or (
                pcd_new.corner_situation == 1 and pcd_pre.corner_situation == 6) or (
                pcd_new.corner_situation == 1 and pcd_pre.corner_situation == 4):
            if pcd_new.is_fea_A_gotten and pcd_pre.is_fea_C_gotten:
                fea_len = min(np.shape(pcd_new.fea_A)[0], np.shape(pcd_pre.fea_C)[0])
                pcd_to_align_new = np.vstack([pcd_to_align_new, pcd_new.fea_A[0:fea_len, :]])
                pcd_to_align_pre = np.vstack([pcd_to_align_pre, pcd_pre.fea_C[0:fea_len, :]])
                if _print_:
                    print("fea_C of new pcd and fea_A of pre pcd:{}".format(fea_len))
                fea_component_new = [fea_len, 0, 0]
                fea_component_pre = [0, 0, fea_len]
                pcd_align_flag = 0
                return pcd_to_align_new, pcd_to_align_pre, pcd_align_flag
        else:
            if pcd_new.is_fea_A_gotten and pcd_pre.is_fea_A_gotten:
                fea_len = min(np.shape(pcd_new.fea_A)[0], np.shape(pcd_pre.fea_A)[0])
                pcd_to_align_new = np.vstack([pcd_to_align_new, pcd_new.fea_A[0:fea_len, :]])
                pcd_to_align_pre = np.vstack([pcd_to_align_pre, pcd_pre.fea_A[0:fea_len, :]])
                if _print_:
                    print("fea_A and fea_A")
                fea_component_new[0] = fea_len
                fea_component_pre[0] = fea_len
            if pcd_new.is_fea_B_gotten and pcd_pre.is_fea_B_gotten:
                fea_len = min(np.shape(pcd_new.fea_B)[0], np.shape(pcd_pre.fea_B)[0])
                pcd_to_align_new = np.vstack([pcd_to_align_new, pcd_new.fea_B[0:fea_len, :]])
                pcd_to_align_pre = np.vstack([pcd_to_align_pre, pcd_pre.fea_B[0:fea_len, :]])
                if _print_:
                    print("fea_B and fea_B")
                fea_component_new[1] = fea_len
                fea_component_pre[1] = fea_len
            if pcd_new.is_fea_C_gotten and pcd_pre.is_fea_C_gotten:
                fea_len = min(np.shape(pcd_new.fea_C)[0], np.shape(pcd_pre.fea_C)[0])
                pcd_to_align_new = np.vstack([pcd_to_align_new, pcd_new.fea_C[0:fea_len, :]])
                pcd_to_align_pre = np.vstack([pcd_to_align_pre, pcd_pre.fea_C[0:fea_len, :]])
                if _print_:
                    print("fea_C and fea_C")
                fea_component_new[2] = fea_len
                fea_component_pre[2] = fea_len
            if np.shape(pcd_to_align_new)[0] < 50:
                pcd_align_flag = 1
            else:
                pcd_align_flag = 0
            return pcd_to_align_new, pcd_to_align_pre, pcd_align_flag
    else:
        print("Extra Feature First")
        pcd_align_flag = 1
        return pcd_to_align_new, pcd_to_align_pre, pcd_align_flag


def is_converge(x, y, scale):
    scale = scale * 0.001
    a = abs(x) < scale
    b = abs(y) < scale
    return a & b


def del_miss(indeces, dist, max_dist, th_rate=0.7):
    th_dist = max_dist * th_rate
    return np.array(indeces[:][np.where(dist[:] < th_dist)[0]])


def icp(pcd_s, pcd_t, max_iterate=50):
    min_len = min(np.shape(pcd_s)[0], np.shape(pcd_t)[0])
    pcd_s_temp = pcd_s[0:min_len, :].astype(np.float32)
    pcd_t_temp = pcd_t[0:min_len, :].astype(np.float32)
    knn = cv2.ml.KNearest_create()
    responses = np.array(range(len(pcd_t_temp[:, 0]))).astype(np.float32)
    knn.train(pcd_s_temp, cv2.ml.ROW_SAMPLE, responses)
    xmove, ymove = 0, 0
    scale_x = np.max(pcd_s_temp[:, 0]) - np.min(pcd_s_temp[:, 0])
    scale_y = np.max(pcd_s_temp[:, 1]) - np.min(pcd_s_temp[:, 1])

    scale = max(scale_x, scale_y)
    for i in range(max_iterate):

        ret, results, neighbours, dist = knn.findNearest(pcd_t_temp, 1)

        indeces = results.astype(np.int32)

        max_dist = sys.maxsize
        indeces = del_miss(indeces, dist, max_dist)

        x_i = pcd_s_temp[indeces, 0]
        y_i = pcd_s_temp[indeces, 1]
        x_j = pcd_t_temp[indeces, 0]
        y_j = pcd_t_temp[indeces, 1]

        dist_x = np.nanmean(x_i - x_j)
        dist_y = np.nanmean(y_i - y_j)

        pcd_t_temp[:, 0] += dist_x
        pcd_t_temp[:, 1] += dist_y
        xmove += dist_x
        ymove += dist_y

        if (is_converge(dist_x, dist_y, scale)):
            break

    return xmove, ymove
