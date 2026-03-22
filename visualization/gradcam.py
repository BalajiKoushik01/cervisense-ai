import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

def generate_gradcam(model, target_layer, img_tensor, original_img_np, class_idx):
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)[0, :]
    visualization = show_cam_on_image(original_img_np, grayscale_cam, use_rgb=True)
    return grayscale_cam, visualization
