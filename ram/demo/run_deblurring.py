import torch
import deepinv as dinv
from deepinv.utils.demo import load_url_image, get_image_url
from ram.models.ram import RAM

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
physics = dinv.physics.BlurFFT(img_size=x_true.shape[1:],
                               noise_model=dinv.physics.GaussianNoise(sigma=0.05),
                               padding="circular",
                               device=device)

# generate measurements
kwargs = generator_physics.step(batch_size=1)
y = physics(x, **kwargs)

img_list = [x, y]
img_titles = ["x", "y"]
dinv.utils.plot(img_list, titles=img_titles)

model = RAM(device=device)

# apply model
with torch.no_grad():
    out = model(y, physics=physics)

img_list.append(out)
img_titles.append("out")
dinv.utils.plot(img_list, titles=img_titles)


