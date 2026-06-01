from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# Load the inpainting model
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

# Load damaged image and mask
image = Image.open("damaged_photo.png").convert("RGB")
mask = Image.open("mask.png").convert("RGB")

# Restoration prompt
prompt = """
restore vintage photograph,
repair scratches and tears,
preserve facial details,
natural lighting,
high-quality photo restoration,
historically accurate appearance
"""

# Generate restored image
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask
).images[0]

# Save output
result.save("restored_photo.png")

print("Photo restored successfully!")