import fire
import pydicom
import cv2
import re
import glob, os
from itertools import product
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def imread_and_imwirte(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
    img = cv2.resize(image, (512, 512),interpolation=cv2.INTER_CUBIC)
    assert img.shape==(512,512)
    cv2.imwrite(dst_path, image)

def combine_lists(a, b, c):
    # Filter out empty lists
    lists = [x for x in (a, b, c) if x]
    # Create combinations of the elements in the non-empty lists
    combinations = list(product(*lists))
    # Convert each combination from tuple to list
    return [list(combination) for combination in combinations]


def main(data_dir, save_dir, nfold=10, seed=42):
    df = pd.read_csv(f'{data_dir}/train_series_descriptions.csv')

    # Unmerge
    study_ids = df['study_id'].unique()
    new_study_ids = []
    new_series_ids = []
    new_series_description = []
    new_alias = []

    for _, si in enumerate(tqdm(study_ids, total=len(study_ids))):
        pdf = df[df['study_id']==si]
        
        axial_t2_series_ids = list(pdf[pdf["series_description"]=="Axial T2"]["series_id"].unique())
        sagittal_t1_series_ids = list(pdf[pdf["series_description"]=="Sagittal T1"]["series_id"].unique())
        sagittal_t2_series_ids = list(pdf[pdf["series_description"]=="Sagittal T2/STIR"]["series_id"].unique())
        
        axial_t2_series = ["Axial T2"] * len(axial_t2_series_ids)
        sagittal_t1_series = ["Sagittal T1"] * len(sagittal_t1_series_ids)
        sagittal_t2_series = ["Sagittal T2/STIR"] * len(sagittal_t2_series_ids)
        
        tmp_series_ids = combine_lists(axial_t2_series_ids, sagittal_t1_series_ids, sagittal_t2_series_ids)
        tmp_series = combine_lists(axial_t2_series, sagittal_t1_series, sagittal_t2_series)
        assert len(tmp_series_ids) == len(tmp_series)
        
        count = 0
        for i, series_ids in enumerate(tmp_series_ids):
            new_study_ids += [si] * len(series_ids)
            new_series_ids += series_ids
            new_series_description += tmp_series[i]
            new_alias += [count] * len(series_ids)
            count += 1
            
    assert len(new_study_ids) == len(new_series_ids) == len(new_series_description) == len(new_alias)
    new_df = pd.DataFrame({
        "study_id": new_study_ids,
        "series_id": new_series_ids,
        "series_description": new_series_description,
        "alias": new_alias,
    })

    # Convert dcm to png
    study_ids = new_df['study_id'].unique()
    desc = list(new_df['series_description'].unique())

    for _, si in enumerate(tqdm(study_ids, total=len(study_ids))):
        pdf = new_df[new_df['study_id']==si]
        for alias in pdf["alias"].unique():
            pdf_alias = pdf[pdf["alias"] == alias]
            for ds in desc:
                ds_ = ds.replace('/', '_').replace(" ", "_")
                pdf_alias_ = pdf_alias[pdf_alias['series_description']==ds]
                assert len(pdf_alias_) == 1
                
                if alias == 0:
                    img_save_dir = f'{save_dir}/images/{si}/{ds_}'
                else:
                    img_save_dir = f'{save_dir}/images/{si}_{alias}/{ds_}'
                os.makedirs(img_save_dir, exist_ok=True)
                
                allimgs = []
                for i, row in pdf_alias_.iterrows():
                    pimgs = glob.glob(f'{data_dir}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
                    pimgs = sorted(pimgs, key=natural_keys)
                    allimgs.extend(pimgs)

                if len(allimgs)==0:
                    print(si, ds, alias, 'has no images')
                    continue
                    
                for j, impath in enumerate(allimgs):
                    dst = f'{img_save_dir}/{j:03d}.png'
                    imread_and_imwirte(impath, dst)

    # KFold
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    unique_ids = new_df['study_id'].unique()
    folds = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_ids)):
        for idx in test_idx:
            folds[unique_ids[idx]] = fold
    new_df['fold'] = new_df['study_id'].map(folds)
    new_df.to_csv("{save_dir}/new_train_series_description.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)