
import torch, time, json
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

prompts = ['A futuristic city at night with neon lights reflecting on wet streets, cyberpunk style', 'A serene mountain lake at sunrise, photorealistic, golden hour lighting', 'A close-up portrait of a robot reading a book, detailed, cinematic', 'Abstract geometric art, vivid colors, high contrast, minimalist', 'A lush forest with rays of sunlight through the canopy, 8k']
n = 10
seed = 42
steps = 20
generator = torch.Generator("cuda").manual_seed(seed)

times = []
for i in range(n):
    prompt = prompts[i % len(prompts)]
    t0 = time.perf_counter()
    pipe(prompt=prompt, num_inference_steps=steps, generator=generator).images[0]
    times.append(time.perf_counter() - t0)

print(json.dumps({
    "per_image_s_mean": round(sum(times)/len(times), 3),
    "per_image_s_min":  round(min(times), 3),
    "per_image_s_max":  round(max(times), 3),
}))
