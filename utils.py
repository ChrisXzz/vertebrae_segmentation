__copyright__ = \
"""
Copyright &copyright © (c) 2021 Inria Grenoble Rhône-Alpes.
All rights reserved.

This source code is to be used for academic research purposes only.
For commercial uses of the code, please send an email to edmond.boyer@inria.fr and sergi.pujades@inria.fr

"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Di Meng"



def mkpath(path):
    import os 

    if not os.path.exists(path):
        os.mkdir(path)


def globalNormalization(x):
    import numpy as np 
    import sys
    from math import sqrt
    """
    Normalize the data by substract mean and then devided by std
    X(i) = x(i)-mean / sqrt(stdˆ2 + e)
    """

    mean = np.mean(x)
    std = np.std(x)

    epsilon = sys.float_info.epsilon

    x_vec = x.flatten().astype(np.float64)
    lengh = len(x_vec)
    for n in range(lengh):
        x_vec[n] = (x_vec[n] - mean)/(sqrt(std**2+epsilon))
    x_norm = np.resize(x_vec, x.shape)

    return x_norm


def read_json_file(file):
    import json

    with open(file, 'r') as f:
        data = f.read()
        jdata = json.loads(data)

    return jdata


def read_annotations_from_json_file(file):
    import json, os 
    import numpy as np 

    with open(file, 'r') as f:
        data = f.read()
        anno = json.loads(data)

    locs = []
    labels = []

    for i in range(len(anno)):
        x = int(anno[i]['X'])
        y = int(anno[i]['Y'])
        z = int(anno[i]['Z'])
        label = int(anno[i]['label'])

        locs.append([x,y,z])
        labels.append(label)

    locs = np.array(locs).astype(np.float)
    labels = np.array(labels)

    resorting_indices = locs[:,1].argsort()
    locs = locs[resorting_indices]
    labels = labels[resorting_indices]

    annotations = {'locations': locs, 'labels': labels}

    return annotations



def read_nifti_file(file):
    import nibabel as nib 

    data = nib.load(file)
    img = data.get_fdata()

    return img 


def save_to_nifti_file(img, save_filename, aff=None):
    import nibabel as nib
    import os
    import numpy as np 

    if aff is not None:
        img = nib.Nifti1Image(img, aff)
    else:
        img = nib.Nifti1Image(img, np.eye(4))

    nib.save(img, save_filename)
    print('saved to {}'.format(save_filename))


def get_size_and_spacing_and_orientation_from_nifti_file(file):
    import nibabel as nib 

    data = nib.load(file)

    size = data.shape

    # read orientation code
    a, b, c = nib.orientations.aff2axcodes(data.affine)
    orientation_code = a+b+c 

    # read voxel spacing 
    header = data.header
    pixdim = header['pixdim']
    spacing = pixdim[1:4]

    aff = data.affine

    return size, spacing, orientation_code, aff


def resampling(nifti_img, spacing, target_shape=None):
    from nilearn.image import resample_img
    import numpy as np 

    new_affine = np.copy(nifti_img.affine)
    new_affine[:3, :3] *= 1.0/spacing

    if target_shape is None:
        target_shape = (nifti_img.shape*spacing).astype(np.int)

    resampled_nifti_img = resample_img(nifti_img, target_affine=new_affine, 
                                                  target_shape=target_shape,
                                                  interpolation='nearest')

    # also return nifti image
    return resampled_nifti_img


def reorienting(img, start_orient_code, end_orient_code):
    import nibabel as nib 

    start_orient = nib.orientations.axcodes2ornt(start_orient_code)
    end_orient = nib.orientations.axcodes2ornt(end_orient_code)

    trans = nib.orientations.ornt_transform(start_orient, end_orient)

    return nib.orientations.apply_orientation(img, trans)


def read_isotropic_pir_img_from_nifti_file(file):
    import nibabel as nib 

    _, spacing, orientation_code, _ = get_size_and_spacing_and_orientation_from_nifti_file(file)

    nifti_img = nib.load(file)

    resampled_nifti_img = resampling(nifti_img, spacing)

    resampled_img = resampled_nifti_img.get_fdata()

    transformed_img = reorienting(resampled_img, orientation_code, 'PIR')

    return transformed_img


def reorient_resample_back_to_original(img, ori_orient_code, spacing, ori_size, ori_aff):
    import nibabel as nib 
    import numpy as np 

    transformed_img = reorienting(img, 'PIR', ori_orient_code)

    nifti_img = nib.Nifti1Image(transformed_img, ori_aff)

    resampled_nifti_img = resampling(nifti_img, 1.0/spacing, ori_size)

    return resampled_nifti_img.get_fdata()


def locations_from_mask(mask):
    import numpy as np 
    from scipy.ndimage.measurements import center_of_mass

    labels = np.unique(mask)[1:]

    assert 0 not in labels

    locations = []

    for label in labels:
        mask_copy = (mask == label)

        x, y, z, = center_of_mass(mask_copy)
        locations.append([x,y,z])

    locations = np.array(locations)
    locations = locations[locations[:,1].argsort()]

    return locations


def annotation_from_mask(mask):
    import numpy as np 
    from scipy.ndimage.measurements import center_of_mass

    labels = np.unique(mask)[1:]

    assert 0 not in labels

    data = []

    for label in labels:
        mask_copy = (mask == label)

        x, y, z, = center_of_mass(mask_copy)

        annotation = dict()

        annotation["label"] = int(label)
        annotation["X"] = float(x)
        annotation["Y"] = float(y)
        annotation["Z"] = float(z)

        data.append(annotation)

    return data 


def write_dict_to_file(data, save_filename):
    import json

    with open(save_filename, 'w') as outfile:
        json.dump(data, outfile)  
    print('annotation saved to {}'.format(save_filename))


def write_result_to_file(pir_mask, ori_orient_code, spacing, ori_size, ori_aff, save_dir, filename):
    import os 

    annotation = annotation_from_mask(pir_mask)

    write_dict_to_file(annotation, os.path.join(save_dir, '{}_ctd.json'.format(filename)))

    mask = reorient_resample_back_to_original(pir_mask, ori_orient_code, spacing, ori_size, ori_aff)

    save_to_nifti_file(mask, os.path.join(save_dir, '{}_seg.nii.gz'.format(filename)), ori_aff)