import torch
import deepinv as dinv
from ram.utils.dataset_utils import get_dataset, get_physics
from ram.models.ram import RAM, finetune

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# choose dataset + inverse problem
# options are 'cryo_em' (denoising real cryo-EM data),
# 'CS_satellite' (Compressed Sensing of satellite images),
# 'demosaicing_satellite' (demosaicing of satellite images),
# 'SPAD' (inpainting and denoising of real SPAD images),

problem = 'SPAD'

# get dataset
batch_size = 4
train_dataset, test_dataset = get_dataset(problem)

# define physics
physics = get_physics(problem, device=device)

# load ram
model = RAM(device=device)

# get one test image
datum = test_dataset[0]
x = datum[0].to(device).unsqueeze(0)
y = datum[1].to(device).unsqueeze(0)
if len(datum) == 3:
    params = datum[2]
    for k in params:
        params[k] = params[k].to(device)
else:
    params = {}

if problem == 'SPAD' or problem == 'cryo_em':
    img_list = [y] # no ground-truth for SPAD or cryo_em
    img_titles = ["Measurement"]
else:
    img_list = [x, y]
    img_titles = ["Ground-truth", "Measurement"]

# apply model
with torch.no_grad():
    physics.update(**params)
    out = model(y, physics)
    zero_shot_psnr = dinv.metric.PSNR()(x, out).mean()

img_list.append(out)
img_titles.append("Zero-Shot")

if problem == 'SPAD' or problem == 'cryo_em':
    noise_loss = 'split'
    transform = None
else:
    noise_loss = 'SURE'
    transform = 'shift'

model = finetune(model, train_dataset, physics, supervised=False, noise_loss=noise_loss, transform=transform, batch_size=batch_size)

# apply model
with torch.no_grad():
    physics.update(**params)
    out2 = model(y, physics)
    finetuned_psnr = dinv.metric.PSNR()(x, out2).mean()


if not problem == 'SPAD' and not problem == 'cryo_em':
    print(f'PSNR zero-shot {zero_shot_psnr:.2f} dB')
    print(f'PSNR finetuned {finetuned_psnr:.2f} dB')

img_list.append(out2)
img_titles.append("Finetuned")
dinv.utils.plot(img_list, titles=img_titles)
