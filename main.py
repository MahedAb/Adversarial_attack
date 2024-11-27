import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def adversarial_attack(model, image, target_class, epsilon=0.03):
    """
    Perform an FGSM adversarial attack to fool the model into classifying the image as the target class.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Set the image requires_grad attribute to True for gradient calculation
    image.requires_grad = True

    # Define the target label
    target = torch.tensor([target_class], dtype=torch.long)

    # Forward pass
    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]
    print(f"Initial class is: {init_pred}")

    if init_pred == target:
        print("Already in the target class!")

    loss = F.nll_loss(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to compute gradients
    loss.backward()

    # Get the gradients of the image
    data_grad = image.grad.data

    # Generate the adversarial example
    perturbed_image = fgsm_attack(image, epsilon, data_grad)

    return perturbed_image


# Example usage
if __name__ == "__main__":
    # Load a pretrained model
    model = resnet18(pretrained=True)

    # Load and preprocess an image
    image_path = "test/image.png"  # Replace with the path to your image
    image = load_image(image_path)

    # Define the target class (e.g., 950 for toaster in ImageNet)
    target_class = 859

    # Perform adversarial attack
    epsilon = 0.25  # Perturbation size
    adv_image = adversarial_attack(model, image, target_class, epsilon)

    output = model(adv_image)
    final_pred = output.max(1, keepdim=True)[1]

    if final_pred == target_class:
        print("Successfully classified as target class!")
    else:
        print(f"Purturbed image classified as: {final_pred}")


    # Save or visualize the adversarial image
    adv_image_np = adv_image.squeeze().detach().cpu().numpy()
    adv_image_np = np.transpose(adv_image_np, (1, 2, 0))  # Convert to HWC
    adv_image_np = (adv_image_np * 255).astype(np.uint8)  # Convert to uint8
    adv_image_pil = Image.fromarray(adv_image_np)
    adv_image_pil.save("output/adversarial_image.png")  # Save the adversarial image
