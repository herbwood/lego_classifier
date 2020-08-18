import torchvision
from torchvision import datasets, transforms
from base import BaseDataLoader

class LegoDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.08660578, 0.08660578, 0.08660578),
                                 (0.17553411, 0.17553411, 0.17553411))])

        self.data_dir = data_dir
        self.dataset = torchvision.datasets.ImageFolder(
            root = self.data_dir,
            transform = transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)