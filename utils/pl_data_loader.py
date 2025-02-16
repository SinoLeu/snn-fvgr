from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
# from timm import create_model
from torchvision import transforms, datasets

## fine-tune resnet
class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_dir: str = './',test_dir: str = './',input_size:int=300,num_class:int = 196):
        super().__init__()
        # self.data_dir = data_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_class

        # Augmentation policy for training set
        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=self.input_size, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=self.input_size),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
        # self.num_classes = 196

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # build dataset '../data/cars/train'
        self.train = datasets.ImageFolder(root=self.train_dir, transform=self.augmentation)
        self.test = datasets.ImageFolder(root=self.test_dir, transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=14)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=14)
