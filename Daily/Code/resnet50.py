import mindspore.dataset.vision.transforms as transforms
from mindspore import nn, Tensor
from mindspore import dtype as mstype
import mindspore as ms
import numpy as np
from PIL import Image
import mindspore.ops as ops
import os

graph = ms.load("resnet50_model.mindir")
resnet_model = nn.GraphCell(graph)
argmax = ops.Argmax(axis=1)
resnet_model.set_train(False)

def preprocess_image_resnet(image_path):
    """Preprocess the input image for prediction with ResNet50."""
    trans = [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ]
    img = Image.open(image_path).convert('RGB')
    for transform in trans:
        img = transform(img)
    img = np.expand_dims(img, axis=0)  # 增加批次维度
    return Tensor(img, mstype.float32)

def predict_with_resnet(image_path):
    """Predict the label for the input image using ResNet50."""
    image = preprocess_image_resnet(image_path)
    output = resnet_model(image)
    return output

def predict_folder_resnet(folder_path):
    """Predict labels for all images in the given folder using ResNet50."""
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
    predictions = {}
    for image_path in images_paths:
        label = predict_with_resnet(image_path)
        predictions[image_path] = label
    return predictions
