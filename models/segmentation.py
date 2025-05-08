import torch
import torchvision.models as models
from torchvision import transforms
from configs.constants import MEAN, STD

class SegmentationModel:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    def _load_model(self):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def segment(self, image):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        return output.argmax(0).cpu().numpy()