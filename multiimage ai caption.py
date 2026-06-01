from transformers import pipeline
from PIL import Image

# Load image captioning model
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# Number of images
num_images = int(input("How many images would you like to caption? "))

captions_report = []

for i in range(num_images):
    image_path = input(f"Enter path for image {i+1}: ")

    try:
        image = Image.open(image_path)

        result = captioner(image)

        caption = result[0]["generated_text"]

        print(f"\nCaption for {image_path}:")
        print(caption)

        captions_report.append(
            f"Image: {image_path}\nCaption: {caption}\n"
        )

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Save report
with open("caption_report.txt", "w") as file:
    file.write("AI IMAGE CAPTION REPORT\n")
    file.write("=" * 30 + "\n\n")

    for item in captions_report:
        file.write(item + "\n")

print("\nReport saved as caption_report.txt")
print(f"Total images processed: {len(captions_report)}")