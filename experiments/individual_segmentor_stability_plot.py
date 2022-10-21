

if __name__ == "__main__":

    from train.data_generator import read_annotation
    import os, argparse
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D

    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--dataset_dir', type=str)
    parser.add_argument('-F', '--dict_folder', type=str, default='individual_segmentor_stability/verse563')
    parser.add_argument('-V', '--vert_labels',  nargs='+', default=[3, 12, 22], help='select vertebrae for the experiment')
    args = parser.parse_args()

    dicts = []

    json_files = [os.path.join(args.dict_folder, 'label_{}.json'.format(l)) for l in args.vert_labels]

    for json_file in json_files:
        ID = os.path.split(json_file)[0].split('/')[-1]
        label_dict =  read_json_file(json_file)

        dicts.append(read_json_file(json_file))

    gt_locations = read_annotation(ID, args.dataset_dir)['locations']


    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title('bassin of attraction of vertebra locations', fontsize=15)

    for i, gt_loc in enumerate(gt_locations):
        ax.scatter(gt_loc[1], gt_loc[0], gt_loc[2], c='black', marker='^')
        ax.text(gt_loc[1], gt_loc[0]+1, gt_loc[2], v_dict[i+1], fontdict={'color': 'black', 'weight': 'bold', 'size': 'x-large'})

    for l_dict in dicts:
        for k in l_dict.keys():
            if k in ['gt_loc', 'gt_msk_size']:
                continue
            data = l_dict[k]

            try:
                con_label = data['converged_label']

            except TypeError:
                continue

            # print(data)
            print('converged label: ', con_label)

            try:
                for waypoint in data['waypoints']:
                    ax.scatter(waypoint[1], waypoint[0], waypoint[2], color=cm_itk(int(con_label)-1), marker='o')
            except TypeError:
                continue

    plt.grid(b=None)
    plt.axis('off')
    plt.show()
