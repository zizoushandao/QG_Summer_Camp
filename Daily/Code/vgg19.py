import mindspore.dataset.vision.transforms as transforms
from mindcv import create_model
from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore import context
import mindspore as ms
import numpy as np
from PIL import Image
import mindspore.ops as ops
import os

# 创建并配置 VGG19 模型
network = create_model('vgg19', pretrained=True)
new_output_layer = nn.Dense(in_channels=network.classifier[6].in_channels, out_channels=10)
network.classifier[6] = new_output_layer
inputs = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
ms.export(network, inputs, file_name="vgg19", file_format="MINDIR")

context.set_context(mode=context.GRAPH_MODE)
graph = ms.load("vgg19.mindir")
vgg_model = nn.GraphCell(graph)
argmax = ops.Argmax(axis=1)
vgg_model.set_train(False)

def preprocess_image_vgg(image_path):
    """Preprocess the input image for prediction with VGG19."""
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

def predict_with_vgg(image_path):
    """Predict the label for the input image using VGG19."""
    image = preprocess_image_vgg(image_path)
    output = vgg_model(image)
    predicted_label = argmax(output).asnumpy()[0]
    return predicted_label

def predict_folder_vgg(folder_path):
    """Predict labels for all images in the given folder using VGG19."""
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
    predictions = {}
    for image_path in images_paths:
        label = predict_with_vgg(image_path)
        predictions[image_path] = label
    return predictions


