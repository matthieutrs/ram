import deepinv as dinv
import torch
import torchvision.transforms as tfs

def get_physics(problem, device='cuda'):
    if problem == 'CS_satellite':
        imsize = (3, 128, 128)
        physics = dinv.physics.CompressedSensing(m=int(128*128/4), channelwise=True, img_shape=imsize, device=device, fast=True, noise_model=dinv.physics.GaussianNoise(sigma=.05))
        physics.load_state_dict(torch.load(f'./finetuning/{problem}/physics0.pt'))
    elif problem == 'SPAD':
        physics = dinv.physics.Inpainting(tensor_size=(1, 256, 256),
                                          mask=None,
                                          noise_model=dinv.physics.PoissonNoise(gain=.2, clip_positive=True),
                                          device=device)
    elif problem == 'cryo_em':
        physics = dinv.physics.Denoising(dinv.physics.PoissonGaussianNoise(sigma=0.98), device=device)
    elif problem == 'demosaicing_satellite':
        imsize = (3, 128, 128)
        physics = dinv.physics.Demosaicing(img_size=imsize, noise_model=dinv.physics.GaussianNoise(.05), device=device)
    else:
        raise NotImplementedError("Problem not implemented")

    return physics

class CryoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, test=False):
        # find all files in folder
        self.dataset = dataset
        self.test = test

    def __len__(self):
        return len(self.dataset) if self.test else 100

    def __getitem__(self, idx):
        input, output = self.dataset[idx % len(self.dataset)]
        if not self.test:
            i, j, h, w = tfs.RandomCrop.get_params(input, output_size=(256, 256))
        else:
            # center crop of (1024, 1024)
            i, j, h, w = 384, 384, 1024, 1024

        input = tfs.functional.crop(input, i, j, h, w)
        output = tfs.functional.crop(output, i, j, h, w)
        return input, output


class PansharpenDataset(torch.utils.data.Dataset):
    def __init__(self):
        data_home = dinv.utils.demo.get_data_home()
        self.dataset = dinv.datasets.NBUDataset(root_dir=data_home, return_pan=True, satellite='worldview-2',
                                                download=True, transform_ms=lambda x: x[[4, 2, 1], ...])
        self.dataset = torch.utils.data.Subset(self.dataset, torch.arange(2, 27))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        MS, pan = self.dataset[idx]
        y = dinv.utils.TensorList([MS, pan])
        x = pan.expand(MS.size(0), pan.size(1), pan.size(1))
        return x, y


def get_dataset(problem):
    import os
    save_dir = f'./finetuning/{problem}'
    if not os.path.exists(save_dir):
        # download from hfhub
        from huggingface_hub import hf_hub_download
        repo_id = "jtachella/finetuning"
        filename = f"{problem}/dinv_dataset0.h5"

        # Download the file
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=f"./finetuning/", repo_type="dataset")
        if problem == 'CS_satellite':
            hf_hub_download(repo_id=repo_id, filename=f"{problem}/physics0.pt", local_dir=f"./finetuning/", repo_type="dataset")

    if problem == 'cryo_em':
        base = dinv.datasets.HDF5Dataset(f'{save_dir}/dinv_dataset0.h5')
        test_dataset = CryoDataset(base, test=True)
        train_dataset = CryoDataset(base)
    elif problem == 'SPAD':
        train_dataset = dinv.datasets.HDF5Dataset(f'{save_dir}/dinv_dataset0.h5', load_physics_generator_params=True)
        test_dataset = train_dataset
    elif problem == 'real_pansharpening':
        train_dataset = PansharpenDataset()
        test_dataset = train_dataset
    else:
        train_dataset = dinv.datasets.HDF5Dataset(f'{save_dir}/dinv_dataset0.h5', load_physics_generator_params=False, train=True)
        test_dataset = dinv.datasets.HDF5Dataset(f'{save_dir}/dinv_dataset0.h5', load_physics_generator_params=False, train=False)

    return train_dataset, test_dataset