import torch
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from ram.models.ram import RAM, finetune

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# load an image
x_true = load_url_image(url=get_image_url("butterfly.png"), img_size=256).to(device)
x = x_true.clone()

# define physics generators
psf_size = 31
generator_motion = dinv.physics.generator.MotionBlurGenerator((psf_size, psf_size), l=2.0, sigma=2.4, device=device)
generator_noise = dinv.physics.generator.SigmaGenerator(sigma_min=0.001, sigma_max=0.2, device=device)
generator_physics = generator_motion + generator_noise

# define physics
physics = dinv.physics.Demosaicing(img_size=x_true.shape[1:], device=device)
physics.set_noise_model(dinv.physics.GaussianNoise(sigma=0.05))

# generate measurements
kwargs = generator_physics.step(batch_size=1)
y = physics(x, **kwargs)

img_list = [x, y]
img_titles = ["Ground-Truth", "Measurement"]
dinv.utils.plot(img_list, titles=img_titles)

model = RAM(device=device)

# apply model
with torch.no_grad():
    out = model(y, physics=physics)
    zero_shot_psnr = dinv.metric.PSNR()(x, out).mean()

img_list.append(out)
img_titles.append("Zero-Shot")

model = finetune(model, y, physics)

# apply model
with torch.no_grad():
    out2 = model(y, physics=physics)
    finetuned_psnr = dinv.metric.PSNR()(x, out2).mean()


print(f'PSNR zero-shot {zero_shot_psnr:.2f} dB')
print(f'PSNR finetuned {finetuned_psnr:.2f} dB')

img_list.append(out2)
img_titles.append("Finetuned")
dinv.utils.plot(img_list, titles=img_titles)