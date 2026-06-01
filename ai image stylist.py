from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter, ImageEnhance
import torch

# Load Stable Diffusion Model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

# Get user input
prompt = input("Enter image prompt: ")
style = input("Choose style (vintage/cartoon/sketch): ").lower()

# Generate image
image = pipe(prompt).images[0]

# Apply style effects
if style == "vintage":
    image = ImageEnhance.Color(image).enhance(0.6)
    image = ImageEnhance.Contrast(image).enhance(1.2)

elif style == "cartoon":
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    image = ImageEnhance.Color(image).enhance(2)

elif style == "sketch":
    image = image.convert("L")
    image = image.filter(ImageFilter.CONTOUR)

# Save image
image.save("styled_image.png")

print("Image saved as styled_image.png")