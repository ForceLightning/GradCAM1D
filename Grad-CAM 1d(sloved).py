import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        
        # Your model

    def forward(self, x):
        
        return p4

def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

def resize_1d(array, shape):
    res = torch.zeros(shape)
    if array.shape[0] >= shape:
        ratio = array.shape[0]/shape
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = torch.flip(res, dims=[0])
        array = torch.flip(array, dims=[0])
        for i in range(array.shape[0]):
            res[int(i/ratio)] += array[i]*(1-(i/ratio-int(i/ratio)))
            if int(i/ratio) != shape-1:
                res[int(i/ratio)+1] += array[i]*(i/ratio-int(i/ratio))
            else:
                res[int(i/ratio)] += array[i]*(i/ratio-int(i/ratio))
        res = torch.flip(res, dims=[0])/(2*ratio)
        array = torch.flip(array, dims=[0])
    else:
        ratio = shape/array.shape[0]
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = torch.flip(res, dims=[0])
        array = torch.flip(array, dims=[0])
        left = 0
        right = 1
        for i in range(shape):
            if left < int(i/ratio):
                left += 1
                right += 1
            if right > array.shape[0]-1:
                res[i] += array[left]
            else:
                res[i] += array[right] * \
                    (i - left * ratio)/ratio+array[left]*(right*ratio-i)/ratio
        res = torch.flip(res, dims=[0])/2
        array = torch.flip(array, dims=[0])
    return res

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation)
        )
        self.handles.append(
            target_layer.register_backward_hook(self.save_gradient)
        )
    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0].cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        """Base class for Class activation mapping.

        Args:
            model (torch.nn.Module): model to inspect
            target_layer (torch.nn.Module): target layer to inspect
            use_cuda (bool, optional): If True, use GPU. Defaults to False.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, batched_input):
        """Forward pass of the model

        Args:
            batched_input (torch.Tensor): signal with shape (batch_size, channels, length)

        Returns:
            torch.Tensor: batched model output with shape (batch_size, num_classes)
        """
        return self.model(batched_input)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise NotImplementedError("This method should be implemented in subclasses")

    def __call__(self, input_tensor, target_category=None):
        """Generates class activation map for a specific category

        Args:
            input_tensor (torch.Tensor): batched signal with shape (batch_size, channels, length)
            target_category (torch.Tensor, optional): batched target with shape (batch_size, ). Defaults to None.

        Returns:
            np.ndarray: CAM for the specified category
        """
        if self.cuda:
            input_tensor = input_tensor.cuda()

        outputs = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = torch.argmax(outputs.cpu().data, dim=0)
        targets = [ClassifierOutputTarget(category) for category in target_category]
        self.model.zero_grad()
        # loss = self.get_loss(output, targets)
        # loss = sum([output[i, target_category[i]] for i in range(output.shape[0])])
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data
        grads = self.activations_and_grads.gradients[-1].cpu().data
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float64)
        weights_reshaped = weights[:, None, None, :]
        activations_reshaped = activations[:, None, :, :]
        cam = torch.matmul(weights_reshaped, activations_reshaped)
        cam = cam.squeeze()
        cam = scipy.signal.resample(cam, input_tensor.shape[1], axis=1)
        cam = np.maximum(cam, 0)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        return heatmap

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"IndexError: {exc_value} in backward pass. Traceback: {exc_tb}")
            return True

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        """GradCAM class for visualizing Convolutional Neural Networks

        Args:
            model (torch.nn.Module): model to inspect
            target_layer (torch.nn.Module): target layer to inspect
            use_cuda (bool, optional):  If True, use GPU. Defaults to False.
        """
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                              target_category,
                              activations, grads):
        if len(grads.shape) == 2:
            return torch.mean(grads, axis=1)
        else:
            return torch.mean(grads, axis=2)

def main():
    model = Net1()
    # model.load_state_dict(torch.load('./data7/parameternn.pt'))
    target_layer = model.p2_6
    net = GradCAM(model, target_layer)
    from settest import Test
    input_tensor = Test.Data[100:101, :]
    input_tensor = torch.tensor(input_tensor, dtype=torch.float64)
    #plt.figure(figsize=(5, 1))
    output = net(input_tensor)
    import scipy.io as scio
    input_tensor = input_tensor.numpy().squeeze()
    dataNew = "G:\\datanew.mat"
    scio.savemat(dataNew, mdict={'cam': output, 'data': input_tensor})

if __name__ == '__main__':
    main()
