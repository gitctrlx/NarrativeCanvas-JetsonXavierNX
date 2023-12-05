import torch
from mmpretrain.apis import get_model
from torch.autograd import Variable

# Ensure that both the model and data are on the same device
device = torch.device('cuda:0')
dummy_input = Variable(torch.randn(1, 3, 224, 224)).to(device)
model_path = '../models/clip-vit-base-p16_openai-pre_3rdparty_in1k.pth'
onnx_output_path = '../onnx/clip-vit-base-p16_openai-pre_3rdparty_in1k.onnx'
model = get_model('vit-base-p16_clip-openai-pre_3rdparty_in1k', pretrained=model_path, device=device)
model.eval()

# Convert to ONNX function
def convert_to_onnx(model, dummy_input, output_path):
    # Ensure that the model is in inference mode
    model.eval()
    # Export model to ONNX using dynamic_ Axes specifies the dynamic batch size
    torch.onnx.export(model, 
                      dummy_input, 
                      output_path, 
                      export_params=True, 
                      opset_version=14, 
                      do_constant_folding=True, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},  # Dynamic batch size
                                    'output': {0: 'batch_size'}})

# Convert to ONNX
convert_to_onnx(model, dummy_input, onnx_output_path)