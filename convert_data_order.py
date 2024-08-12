import os
import ast
import cv2
import fire
import pydicom
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold


def imread_and_imwirte(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    #image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    #assert image.shape==(512,512)
    cv2.imwrite(dst_path, image.astype('uint8'))

def main(data_dir, save_dir, nfold=5, seed=42, convert_data=True):
    df = pd.read_csv(f'{data_dir}/new_train_series_descriptions.csv')
    df_label = pd.read_csv(f'{data_dir}/train.csv')

    # KFold based on study_id
    kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    study_ids = df['study_id'].unique()
    folds = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(study_ids)):
        for idx in test_idx:
            folds[study_ids[idx]] = fold
    df['fold'] = df['study_id'].map(folds)
    df.to_csv(f"{save_dir}/train_series_description_{nfold}folds.csv", index=False)

    df_label['fold'] = df_label['study_id'].map(folds)
    df_label.to_csv(f"{save_dir}/train_{nfold}folds.csv", index=False)

    # Convert dcm to png
    if convert_data:
        study_ids = df['study_id'].unique()
        desc = list(df['series_description'].unique())

        for _, si in enumerate(tqdm(study_ids, total=len(study_ids))):
            pdf = df[df['study_id']==si]
            for ds in desc:
                ds_ = ds.replace('/', '_').replace(" ", "_")
                pdf_ = pdf[pdf['series_description']==ds]
                assert len(pdf_) < 2

                img_save_dir = f'{save_dir}/images/{si}/{ds_}'
                os.makedirs(img_save_dir, exist_ok=True)

                if len(pdf_) == 0:
                    print(si, ds, 'has no images')
                    continue
                else:
                    for i, row in pdf_.iterrows():
                        allimgs = [f'{data_dir}/train_images/{file_path}' for file_path in ast.literal_eval(row["file_paths"])]

                for j, impath in enumerate(allimgs):
                    dst = f'{img_save_dir}/{j:03d}.png'
                    imread_and_imwirte(impath, dst)


if __name__ == "__main__":
    fire.Fire(main)