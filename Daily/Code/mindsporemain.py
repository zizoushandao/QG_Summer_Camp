import os
from mindspore import nn, Tensor
from mindspore import dtype as mstype
from mindspore import context
import mindspore as ms
import numpy as np
from PIL import Image
import mindspore.dataset.vision.transforms as transforms
from mindspore import ops

# 设置MindSpore的运行上下文
context.set_context(mode=context.GRAPH_MODE)

# 定义 argmax 操作
argmax = ops.Argmax(axis=1)

class CustomPreprocessingLayer(nn.Cell):
    """自定义层，将 [1, 10] 的特征向量转换为 [1, 3, 224, 224] 的张量。"""
    def __init__(self):
        super(CustomPreprocessingLayer, self).__init__()
        self.fc = nn.Dense(10, 3 * 224 * 224)
        self.reshape = ops.Reshape()

    def construct(self, x):
        x = self.fc(x)
        x = self.reshape(x, (1, 3, 224, 224))
        return x
custom_preprocessing_layer=CustomPreprocessingLayer()
def predict_combined(model1, model2, image_path):
    """串联预测输入图像的标签。"""
    model1_output = predict(model1, image_path)
    transformed_output = custom_preprocessing_layer(model1_output)
    model2_output = model2(transformed_output)
    predicted_label = argmax(model2_output).asnumpy()[0]
    return predicted_label

def preprocess_image(image_path):
    """预处理输入图像以进行预测。"""
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

def load_model(model_path):
    """加载模型并返回模型实例。"""
    graph = ms.load(model_path)
    model = nn.GraphCell(graph)
    model.set_train(False)
    return model

def get_tensor_shape(model, image_path):
    """获取模型输出的张量形状。"""
    image = preprocess_image(image_path)
    output = model(image)
    return output.shape

def predict(model, image_path):
    """使用给定模型预测输入图像的标签。"""
    image = preprocess_image(image_path)
    output = model(image)
    return output  # 返回模型的输出



def predict_folder(folder_path, model, combined_model=None, model_choice=None):
    """预测给定文件夹中所有图像的标签。"""
    images_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))]
    predictions = {}
    for image_path in images_paths:
        if model_choice == "3" and combined_model is not None:
            label = predict_combined(model, combined_model, image_path)
        else:
            output = predict(model, image_path)
            label = argmax(output).asnumpy()[0]
        predictions[image_path] = label
    return predictions

def main():
    model1_path = input("请输入第一个模型文件路径: ")
    model2_path = input("请输入第二个模型文件路径: ")

    model1 = load_model(model1_path)
    model2 = load_model(model2_path)

    while True:
        model_choice = input("请选择模型 (1: the first model, 2: the second model, 3: combined, 4: 退出): ")
        if model_choice == "4":
            break

        input_path = input("请输入图像文件或文件夹路径: ")

        if os.path.isfile(input_path):
            if model_choice == "1":
                output = predict(model1, input_path)
                label = argmax(output).asnumpy()[0]
            elif model_choice == "2":
                output = predict(model2, input_path)
                label = argmax(output).asnumpy()[0]
            elif model_choice == "3":
                label = predict_combined(model1, model2, input_path)
            print(f"图像 {input_path} 的预测标签是: {label}")
        elif os.path.isdir(input_path):
            predictions = predict_folder(input_path, model1 if model_choice == "1" else model2, model2, model_choice)
            for image_path, label in predictions.items():
                print(f"图像 {image_path} 的预测标签是: {label}")
        else:
            print("输入路径无效，请输入有效的图像文件或文件夹路径。")

if __name__ == "__main__":
    main()
