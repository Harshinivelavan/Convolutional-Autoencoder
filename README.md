# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.


## DESIGN STEPS

### STEP 1:


Import necessary libraries including PyTorch, torchvision, and matplotlib.


### STEP 2:

Load the MNIST dataset with transforms to convert images to tensors.


### STEP 3:

Add Gaussian noise to training and testing images using a custom function.

### STEP 4:
Define the architecture of a convolutional autoencoder:

Encoder: Conv2D layers with ReLU + MaxPool

Decoder: ConvTranspose2D layers with ReLU/Sigmoid

### STEP 5:
Initialize model, define loss function (MSE) and optimizer (Adam).

### STEP 6:
Train the model using noisy images as input and original images as target.

### STEP 7:
Visualize and compare original, noisy, and denoised images.

## PROGRAM

### Name:Harshini.V
### Register Number:212224040109

~~~
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # (1,28,28) -> (32,14,14)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32,14,14) -> (64,7,7)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64,7,7) -> (32,14,14)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # (32,14,14) -> (1,28,28)
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: P PARTHIBAN")
    print("Register Number: 212223230145")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

~~~


### DATASET

<img width="585" height="355" alt="image" src="https://github.com/user-attachments/assets/3f118ea3-9b54-4d9e-9475-1a856ff0617c" />


## OUTPUT

### Model Summary

<img width="789" height="468" alt="image" src="https://github.com/user-attachments/assets/36019aad-f7ce-4262-a860-f6a08ce035c7" />



### Original vs Noisy Vs Reconstructed Image


<img width="447" height="392" alt="image" src="https://github.com/user-attachments/assets/3c768469-cd22-49bf-be06-df08b9bd1f46" />




## RESULT


The convolutional autoencoder was successfully trained to remove noise from MNIST images, effectively reconstructing clean and clear outputs from noisy inputs.
