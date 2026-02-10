from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class OxfordPetsSegmentation(Dataset):

    def __init__(self, root='../dataset', split='train', image_size=128):
        self.dataset = OxfordIIITPet(root=root, download=True, target_types='segmentation')
        self.images = self.dataset._images
        self.masks = self.dataset._segs
        total = len(self.images)
    
        if split == 'train':
            self.images = self.images[:int(0.8 * total)]
            self.masks = self.masks[:int(0.8 * total)]
        
        else:
            self.images = self.images[int(0.8 * total):]
            self.masks = self.masks[int(0.8 * total):]
        
        self.img_transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.mask_transform = T.Compose([T.Resize((image_size, image_size), interpolation=Image.NEAREST), T.PILToTensor()])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        img = self.img_transform(img)
        mask = self.mask_transform(mask).squeeze(0) - 1
        return img, mask.long()


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=3):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
    
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_classes, 1)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final(dec1)



train_data = OxfordPetsSegmentation(split='train')
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
model = UNet(in_channels=3, out_classes=3).to('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
        images, masks = images.to('cuda' if torch.cuda.is_available() else 'cpu'), masks.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# plot loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

# visualize some predictions
test_data = OxfordPetsSegmentation(split='test')
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
x, y = next(iter(test_loader))
x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    model.eval()
    pred = torch.argmax(model(x), dim=1).cpu()

# plot
fig, axes = plt.subplots(3, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(x[i].cpu().permute(1, 2, 0))
    axes[0, i].set_title('Input Image')
    axes[0, i].axis('off')

    axes[1, i].imshow(y[i], cmap='gray')
    axes[1, i].set_title('Ground Truth')
    axes[1, i].axis('off')

    axes[2, i].imshow(pred[i], cmap='gray')
    axes[2, i].set_title('Predicted Mask')
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()