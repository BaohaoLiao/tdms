import fire
import pydicom
import glob, os
import pandas as pd
import cv2
from tqdm import tqdm
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text)]

def imread_and_imwirte(src_path, dst_path):
    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    #image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
    #img = cv2.resize(image, (512, 512),interpolation=cv2.INTER_CUBIC)
    #assert img.shape==(512,512)
    cv2.imwrite(dst_path, image)

def main(root_dir, save_dir):
    dfc = pd.read_csv(f'{root_dir}/train_label_coordinates.csv')
    df = pd.read_csv(f'{root_dir}/train_series_descriptions.csv')
    desc = list(df['series_description'].unique())
    st_ids = df['study_id'].unique()

    for idx, si in enumerate(tqdm(st_ids, total=len(st_ids))):
        pdf = df[df['study_id']==si]
        for ds in desc:
            ds_ = ds.replace('/', '_').replace(' ', '_')
            pdf_ = pdf[pdf['series_description']==ds]
            os.makedirs(f'{save_dir}/{si}/{ds_}', exist_ok=True)
            allimgs = []
            for i, row in pdf_.iterrows():
                pimgs = glob.glob(f'{root_dir}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
                pimgs = sorted(pimgs, key=natural_keys)
                allimgs.extend(pimgs)
                
            if len(allimgs)==0:
                print(si, ds, 'has no images')
                continue

            for j, impath in enumerate(allimgs):
                dst = f'{save_dir}/{si}/{ds_}/{j:03d}.png'
                imread_and_imwirte(impath, dst)


if __name__ == "__main__":
    fire.Fire(main)