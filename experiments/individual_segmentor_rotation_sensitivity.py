

if __name__ == "__main__":

    from train.data_generator import read_image_and_mask, read_annotation
    import os, argparse
    import pandas as pd 
    import numpy as np 
    from metrics import compute_dice
    from utils import mkpath
    from segment_vertebra import per_location_rotated_segmentor

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_dir', type=str)
    parser.add_argument('-S', '--spine_id', type=str, default='verse563')
    parser.add_argument('-V', '--vert_labels',  nargs='+', default=[3, 12, 22], help='select vertebrae for the experiment')
    parser.add_argument('-R', '--save_dir', type=str, help='the folder to save the results')
    args = parser.parse_args()

    model_file = '../models/segmentor_vertebra.pth'

    mkpath(args.save_dir)
    save_filename = os.path.join(args.save_dir, args.spine_id + '.csv')

    pir_img, pir_msk = read_image_and_mask(args.spine_id, args.dataset_dir, pir_orientation=True)
    annotations = read_annotation(args.spine_id, args.dataset_dir)
    locations, labels = annotations['locations'], annotations['labels']

    score_mat = dict()

    for label, loc in zip(labels, locations):
        
        if label not in args.vert_labels:
            continue

        gt_msk = (pir_msk == label)

        score_mat[str(label)] = []

        for angle in range(-180, 180+10, 10):

            pred_loc, pred_mask = per_location_rotated_segmentor(loc, angle, pir_img, model_file)

            dsc_score = compute_dice(pred_mask, gt_msk)
            print('label: {}, angle: {}, dsc: {}'.format(label, angle, dsc_score))

            score_mat[str(label)].append(dsc_score)

    col = [str(i) for i in range(-180, 180+10, 10)]
    data = pd.DataFrame.from_dict(score_mat, orient='index', columns=col)
    print(data)
    data.to_csv(save_filename)