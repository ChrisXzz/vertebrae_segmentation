def location_in_the_view(location, vol_size):

    location[location < 0] = 0

    if (location <= vol_size).all():
        return location
    else:
        for i in range(3):
            location[i] = vol_size[i] if location[i] > vol_size[i] else location[i]
        return location


def compute_dist(loc1, loc2):
    import numpy as np 
    x1, y1, z1 = loc1[:]
    x2, y2, z2 = loc2[:]

    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def per_location_from_segmentation_iter(loc, pir_img, model_file):

    import numpy as np 
    from segment_vertebra import per_location_refiner_segmentor
    from consistency_loop import loc_and_msk_convergence, masks_overlapping_pct

    loc_list = []
    msk_list = []

    loc_list.append(loc)
    msk_list.append(None)

    loc2pre_dist = []
    msk2pre_overlap = []

    it = 0
    while True:
        it += 1

        x, y, z = loc[:]
        loc, mask = per_location_refiner_segmentor(x, y, z, pir_img, model_file)

        if loc is None:
            break

        if it > 10:
            break

        loc_list.append(loc)
        msk_list.append(mask)

        if len(loc_list) > 1:

            loc_pre = loc_list[-2]
            dist_to_pre = compute_dist(loc_pre, loc)
            loc2pre_dist.append(dist_to_pre)

        if len(msk_list) > 1:

            msk_pre = msk_list[-2]
            pct_to_pre = masks_overlapping_pct(mask, msk_pre)
            msk2pre_overlap.append(pct_to_pre)

        if loc_and_msk_convergence(loc_list, loc2pre_dist, msk2pre_overlap):
            break

    return loc_list, msk_list[-1]


def vert_converged_to(gt_locations, gt_labels, con_loc):

    import numpy as np 

    dists = []

    for (gt_loc, gt_label) in zip(gt_locations, gt_labels):

        d = compute_dist(gt_loc, con_loc)
        dists.append([d, gt_label])

    dists = np.array(dists)

    con_label = dists[np.argmin(dists[:,0]), 1]
    dist2gt = dists[np.argmin(dists[:,0]), 0]

    return con_label, dist2gt



if __name__ == "__main__":

    from train.data_generator import read_image_and_mask, read_annotation
    import os, argparse
    import numpy as np 
    from utils import mkpath, write_dict_to_file

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_dir', type=str)
    parser.add_argument('-S', '--spine_id', type=str, default='verse563')
    parser.add_argument('-V', '--vert_labels',  nargs='+', default=[3, 12, 22], help='select vertebrae for the experiment')
    parser.add_argument('-R', '--save_dir', type=str, help='folder to save the results', default='individual_segmentor_stability/')
    args = parser.parse_args()

    mkpath(args.save_dir)
    save_folder = os.path.join(args.save_dir, args.spine_id)
    mkpath(save_folder)

    model_file = '../models/segmentor_vertebra.pth'

    # read the spine and vertebrae 
    pir_img, pir_msk = read_image_and_mask(args.spine_id, args.dataset_dir, pir_orientation=True)
    anno = read_annotation(args.spine_id, args.dataset_dir)
    locations = anno['locations']
    labels = anno['labels']    

    for (loc, label) in zip(locations, labels):

        data = dict()

        if label not in args.vert_labels:
            continue 
        print('label: ', label)

        save_filename = os.path.join(save_folder, 'label_{}.json'.format(str(label)))
        if os.path.exists(save_filename):
            continue

        # log the gt info
        gt_loc = loc
        gt_mask = (pir_msk == label).astype(np.int)

        data['gt_loc'] = gt_loc.tolist()
        data['gt_msk_size'] = np.count_nonzero(gt_mask==1)

        # items to log
        loc_waypoints_positive = []
        loc_waypoints_negative = []
        loc_init_dist_positive = []
        loc_init_dist_negative = []
        pred_msk_size = []

        # create a bbox around the location
        search_area = np.array([30, 30, 30])
        bbox_top_left = loc - search_area
        bbox_right_down = loc + search_area

        bbox_top_left = location_in_the_view(bbox_top_left, pir_img.shape)
        bbox_right_down = location_in_the_view(bbox_right_down, pir_img.shape)

        tl_x, tl_y, tl_z = bbox_top_left[:]
        rd_x, rd_y, rd_z = bbox_right_down[:]

        tl_x, tl_y, tl_z = int(tl_x), int(tl_y), int(tl_z)
        rd_x, rd_y, rd_z = int(rd_x), int(rd_y), int(rd_z)

        # stride set to 10
        step = 10

        # run location-segmentation iteration on each sample and log the trajactory
        count = 0
        for i in range(tl_x, rd_x+step, step):
            for j in range(tl_y, rd_y+step, step):
                for k in range(tl_z, rd_z+step, step):

                    count += 1
                    data[str(count)] = dict()

                    seed_loc = np.array([i, j, k])
                    print('seed loc: ', seed_loc)

                    waypoints, pred_msk = per_location_from_segmentation_iter(seed_loc, pir_img, model_file)

                    if pred_msk is None:
                        continue 

                    pred_msk_size.append(np.count_nonzero(pred_msk==1))


                    converged_label, dist2gt = vert_converged_to(locations, labels, waypoints[-1])
                    print('converged to label {} with distance {}'.format(converged_label, dist2gt))

                    data[str(count)]['waypoints'] = [i.tolist() for i in waypoints]
                    data[str(count)]['converged_label'] = converged_label
                    if converged_label == label:
                        data[str(count)]['converged'] = 1
                    else:
                        data[str(count)]['converged'] = 0

                    print(data)

        data['pred_msk_size'] = np.mean(np.array(pred_msk_size))

        data = dict((k,v) for k,v in data.items() if v)

        print(data)

        write_dict_to_file(data, save_filename)



