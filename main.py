import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import argparse
import matplotlib.pyplot as plt


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

    if init_pred == torch.tensor([target_class], dtype=torch.long):
        print("Already in the target class!")

    loss = nn.CrossEntropyLoss()(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to compute gradients
    loss.backward()

    # Get the gradients of the image
    data_grad = -image.grad.data

    # Generate the adversarial example
    perturbed_image = fgsm_attack(image, epsilon, data_grad)

    return perturbed_image


def adversarial_attack_iter(model, image, target_class, epsilon=0.001, max_num_iter=10):
    """
    Perform an FGSM adversarial attack to fool the model into classifying the image as the target class. In this function
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Set the image requires_grad attribute to True for gradient calculation
    image.requires_grad = True

    # Define the target label
    target = torch.tensor([target_class], dtype=torch.long)

    output = model(image)
    init_pred = output.max(1, keepdim=True)[1]
    print(f"Initial class is: {init_pred}")

    if init_pred == target:
        print("Already in the target class!")
        return image

    num_iter = 0
    while num_iter < max_num_iter:
        # Forward pass
        output = model(image)

        loss = nn.CrossEntropyLoss()(output, target)
        print(f" Loss at this iteration is: {loss}")
        # Zero all existing gradients
        model.zero_grad()

        # Backward pass to compute gradients
        loss.backward()

        # Get the gradients of the image
        data_grad = -image.grad.data
        # Generate the adversarial example
        perturbed_image = fgsm_attack(image, epsilon, data_grad)

        curr_output = model(perturbed_image)
        adv_pred = curr_output.max(1, keepdim=True)[1]

        if adv_pred == target_class:
            break
        else:
            print(f"Iteration {num_iter}")
            print(f"current class is: {adv_pred}")

        image = perturbed_image.detach()
        image.requires_grad = True
        num_iter += 1

    return perturbed_image


# Example usage
if __name__ == "__main__":
    # Load a pretrained model
    model = resnet18(pretrained=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='fgsm', help='one of the two methods')
    args = parser.parse_args()

    # Load and preprocess an image
    img = Image.open('test/panda.png')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Define the target class
    target_class = 100

    # Perform adversarial attack
    epsilon = 0.005  # Perturbation size
    if args.method == 'fgsm':
        adv_image = adversarial_attack(model, img_tensor, target_class, epsilon)
    elif args.method == 'ifgsm':
        adv_image = adversarial_attack_iter(model, img_tensor, target_class, epsilon)
    else:
        raise ValueError("Invalid method")

    adv_image = adv_image.squeeze(0).permute(1, 2, 0).detach().numpy()
    plt.imsave("output/output_image.png", adv_image)

    # adv_image_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(adv_image)
    output = model(transform(adv_image).unsqueeze(0))
    final_pred = output.max(1, keepdim=True)[1]

    if final_pred == target_class:
        print("Successfully classified as target class!")
    else:
        print(f"Purturbed image classified as: {final_pred}")

