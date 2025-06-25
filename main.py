from collections import defaultdict
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


def calculate_channel_attributions(img):
    # Forward pass up to last layer
    x = model.conv1(img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    last_layer_activations = model.layer4(x)  # (1, 2048, 7, 7)

    # Compute avgpool and FC manually
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    pooled = avgpool(last_layer_activations)
    flattened = pooled.view(pooled.size(0), -1)
    logits = model.fc(flattened)  # (1, 1000)

    # Select target logit
    target_logit = logits[0, logits.argmax(dim=1)]

    # Compute gradient of target logit wrt last layer activations
    grads = torch.autograd.grad(target_logit, last_layer_activations)[0]

    # Sum gradients across spatial dimensions
    attributions = grads.sum(dim=(2, 3)).abs().squeeze()

    return attributions


# Import ResNet, set to evaluation mode
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()


# Load validation split of the ImageNet dataset, apply transforms, init data loader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data = datasets.ImageNet(
    root="./",
    split="val",
    transform=transform
)
loader = DataLoader(
    data,
    batch_size=1,
    shuffle=True,
)


n_labels = 1000
n_channels = 2048

# Create variables for saving stats
label_counts = [0 for l in range(n_labels)]
important_channels = [defaultdict(int) for l in range(n_labels)]
poly = defaultdict(int)

# For each image
for idx, (img, label) in tqdm(enumerate(loader)):
    # Increase label count
    label_counts[label] += 1

    if idx >= 10000:
         break  # use first 10000 images

    # Get attributions
    img_attributions = calculate_channel_attributions(img)

    # Find important channels for the image
    # -> Attribution >= 2% of total
    sum_attributions = sum(img_attributions)
    for c in range(n_channels):
        if img_attributions[c] >= 0.02 * sum_attributions:
            important_channels[label][c] += 1


# Find polysemantic channels
# -> Important in >= 75% images with a specified label
for l in range(n_labels):
    if important_channels[l]:
        for c in range(n_channels):
            if important_channels[l][c] >= 0.75 * label_counts[l]:
                poly[c] += 1


print(sorted([(c, poly[c]) for c in poly], key=lambda x:x[1], reverse=True))