__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def load_models():

    model_file_seg_binary = 'models/segmentor_spine.pth'
    model_file_seg_idv = 'models/segmentor_vertebra.pth'

    model_file_loc_sag = 'models/locator_sagittal.pth'
    model_file_loc_cor = 'models/locator_coronal.pth'

    id_group_model_file = 'models/classifier_group.pth'
    id_cer_model_file = 'models/classifier_cervical.pth'
    id_thor_model_file = 'models/classifier_thoracic.pth'
    id_lum_model_file = 'models/classifier_lumbar.pth'


    return {'seg_binary': model_file_seg_binary, 'seg_individual': model_file_seg_idv, 
            'loc_sagittal': model_file_loc_sag, 'loc_coronal': model_file_loc_cor, 
            'id_group': id_group_model_file, 'id_cervical': id_cer_model_file, 
            'id_thoracic': id_thor_model_file, 'id_lumbar': id_lum_model_file}


if __name__ == "__main__":

    import argparse, os, glob, sys
    from utils import mkpath, read_isotropic_pir_img_from_nifti_file
    import numpy as np 
    import torch
    torch.set_grad_enabled(False)


    parser = argparse.ArgumentParser(description='Run pipeline on a single CT scan.')
    parser.add_argument('-D', '--data_file', type=str, help='a CT scan in nifti format')
    parser.add_argument('-S', '--save_folder',  default='-1', type=str, help='folder to save the results')
    parser.add_argument('-F', '--force_recompute', action='store_true', help='set True to recompute and overwrite the results')
    args = parser.parse_args()


    ### results saving locations
    save_folder = args.save_folder
    if save_folder != '-1':
        mkpath(save_folder)
    else:
        current_path = os.path.abspath(os.getcwd())
        save_folder = os.path.join(current_path, 'results')
        mkpath(save_folder)


    ### load trained models
    models = load_models()

    ### check results existence
    scanname = os.path.split(args.data_file)[-1].split('.')[0]
    print(' ... checking: ', scanname)

    if os.path.exists(os.path.join(save_folder, '{}_seg.nii.gz'.format(scanname))) and not args.force_recompute:
        sys.exit(' ... {} result exists, not overwriting '.format(scanname))

    print(' ... starting to process: ', scanname)



    # =================================================================
    # Load the CT scan 
    # =================================================================

    ### TODO: data I/O for other formats

    try:
        pir_img = read_isotropic_pir_img_from_nifti_file(args.data_file)
    except ImageFileError:
        sys.exit('The input CT should be in nifti format.')

    print(' ... loaded CT volume in isotropic resolution and PIR orientation ')

    # =================================================================
    # Spine binary segmentation
    # =================================================================

    from segment_spine import binary_segmentor
    binary_mask = binary_segmentor(pir_img, models['seg_binary'], mode='overlap', norm=False)

    print(' ... obtained spine binary segmentation ')

    # =================================================================
    # Initial locations
    # =================================================================

    locations = []
    from locate import locate 
    locations = locate(pir_img, models['loc_sagittal'], models['loc_coronal'])

    print(' ... obtained {} initial 3D locations '.format(len(locations)))

    # =================================================================
    # Consistency circle - Locations refine - multi label segmentation 
    # =================================================================

    from consistency_loop import consistency_refinement_close_loop

    multi_label_mask, locations, labels, loc_has_converged = consistency_refinement_close_loop(locations, pir_img, binary_mask,
                                                                        models['seg_individual'],
                                                                        models['id_group'], models['id_cervical'],
                                                                        models['id_thoracic'], models['id_lumbar'])            

    print(' ... obtained PIR multi label segmentation ')

    # =================================================================
    # Save the result in original format
    # =================================================================

    from utils import get_size_and_spacing_and_orientation_from_nifti_file, write_result_to_file
    ori_size, ori_spacing, ori_orient_code, ori_aff = get_size_and_spacing_and_orientation_from_nifti_file(args.data_file)

    write_result_to_file(multi_label_mask, ori_orient_code, ori_spacing, ori_size, ori_aff, save_folder, scanname)