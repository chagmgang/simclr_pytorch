import torch
import model
import nt_xent_loss
import dataset_wrapper

from torchvision import transforms

def train():
    input_shape = (112, 112)
    batch_size = 3
    num_workers = 4
    temperature = 0.5
    use_cosine_similarity = True

    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    data_transforms = transforms.Compose([
        transforms.Resize(size=input_shape[0]),
        transforms.CenterCrop(size=input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(45),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        dataset_wrapper.GaussianBlur(kernel_size=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    data_augmentation = dataset_wrapper.SimCLRDataTransform(data_transforms)
    dataset = dataset_wrapper.SimCLRDataset(
        image_root='data',
        transforms=data_augmentation)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simclr = model.Model()
    simclr.to(device)

    nt_xent_criterion = nt_xent_loss.NTXentLoss(
            device=device, batch_size=batch_size,
            temperature=temperature, use_cosine_similarity=use_cosine_similarity)

    optimizer = torch.optim.Adam(simclr.parameters(), 3e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    for epoch in range(100):

        total_loss = 0
        for i, (xis, xjs) in enumerate(train_loader):
            xis, xjs = xis.to(device), xjs.to(device)

            optimizer.zero_grad()
            ris, zis = simclr(xis)
            rjs, zjs = simclr(xjs)

            zis = torch.nn.functional.normalize(zis, dim=1)
            zjs = torch.nn.functional.normalize(zjs, dim=1)

            loss = nt_xent_criterion(zis, zjs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('----------------')
        print(f'epoch : {epoch}')
        print(f'loss  : {total_loss / len(train_loader)}')
        print(f'lr    : {scheduler.get_lr()[0]}')

        scheduler.step()

if __name__ == '__main__':
    train()