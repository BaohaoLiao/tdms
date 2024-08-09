from data_loader.sampling_dataset import RSNASamplingDataset
from data_loader.sampling_dataset_noalias import RSNASamplingDataset as RSNASamplingDatasetNoAlias

DATASETS = {
    "sampling": RSNASamplingDataset,
    "sampling_noalias": RSNASamplingDatasetNoAlias,
}
