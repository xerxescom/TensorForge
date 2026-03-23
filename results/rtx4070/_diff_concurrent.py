
import torch, time, json
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

t_end = time.time() + 60
n_images = 0
total_steps = 0
steps = 10

while time.time() < t_end:
    pipe(prompt="a red apple", num_inference_steps=steps)
    n_images += 1
    total_steps += steps

print(json.dumps({
    "n_images": n_images,
    "total_steps": total_steps,
    "it_per_s": round(total_steps / 60, 3),
}))
