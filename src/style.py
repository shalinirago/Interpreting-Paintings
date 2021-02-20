import torch
import numpy as np
from PIL import Image


# Class to preprocess image and predict output
class StyleClassification:
    def __init__(self, model_path):
        """
        Performs prediction of style of input painting
        :param style_model: Given trained VGG-16 network that classifies styles of paintings
        :returns: Predicted style and probability of classification
        """
        # Model saved - define model path
        # Loading trained model
        self.model = torch.load(model_path)
        self.model_classes = ('Abstract Expressionism', 'Baroque', 'Cubism', 'Expressionism', 'Fauvism',
                              'Impressionism', 'Minimalism', 'Post Impressionism', 'Renaissance')

    def process_image(self, im_path):
        """
        Preprocess image
        :param im_path: Input image file path
        :returns img_tensor: Processed Image Tensor
        """
        # Process an image path into a PyTorch tensor
        image = Image.open(im_path)
        # Resize
        img = image.resize((256, 256))
        # Center crop
        width = 256
        height = 256
        new_width = 224
        new_height = 224
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))
        # Convert to numpy, transpose color dimension and normalize
        img = np.array(img).transpose((2, 0, 1)) / 256
        # Standardization
        means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img = img - means
        img = img / stds
        img_tensor = torch.Tensor(img)

        return img_tensor

    def predict(self, img_path, top_k=1):
        """
        Make a prediction for an image using a trained model
        :param img_path: Given input image file path
        :param top_k: number of top predictions to return
        :type top_k: int
        :returns top_p: The top_k predicted probabilities
        :returns top_classes: The top_k predicted classes the input image belongs to
        :rtype top_p: float
        :rtype top_classes: string
        """
        # Convert to pytorch tensor
        img_tensor = self.process_image(img_path)
        # Resize
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
        # Set to evaluation
        with torch.no_grad():
            self.model.eval()
            out = self.model(img_tensor)
            ps = torch.nn.functional.softmax(out, dim=1)
            # print("ps:", ps)
            # Find the top_k predictions
            top_k, top_class = ps.topk(top_k, dim=1)
            # print("top_k:", top_k)
            # print("top_class:", top_class)
            # Extract the actual classes and probabilities
            top_classes = [self.model_classes[class_] for class_ in top_class.cpu().numpy()[0]]
            top_p = top_k.cpu().numpy()[0]

            return top_p, top_classes
