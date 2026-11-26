import torch.nn.functional as F

def grad_cam(model, x, target=1):
    '''
    Grad-CAM for class `target` (1 = defective) from ResNet18's last conv block (layer4).
    Returns an HxW numpy array in [0, 1] at the input resolution: where the model looked.
    '''
    saved = {}

    def hook(module, inputs, output):
        saved["acts"] = output
        output.register_hook(lambda grad: saved.update(grads=grad))

    handle = model.model.layer4.register_forward_hook(hook)
    try:
        model.zero_grad()
        model(x)[0, target].backward()
    finally:
        handle.remove()

    weights = saved["grads"].mean(dim=(2, 3), keepdim=True)  # how much each channel pushes toward the class
    cam = F.relu((weights * saved["acts"]).sum(dim=1, keepdim=True))  # keep only evidence *for* the class
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear")[0, 0]  # 7x7 -> 224x224
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
