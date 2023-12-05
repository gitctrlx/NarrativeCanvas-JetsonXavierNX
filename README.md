# Narrative Canvas: Image-Inspired Storytelling

Narrative Canvas, also known as "Language Within the Paintings," is the very essence of this project. Here, each canvas is not merely a combination of colors and lines but a collection of untold stories waiting to be discovered. Artists unleash their imaginations onto the canvas, and every stroke and every brushstroke carries profound emotions and a unique perspective. These artworks, akin to poems without words, quietly narrate their own tales.

This project has successfully implemented image inference tasks, text generation tasks, and image generation tasks on the Jetson development board. It utilizes TensorRT for accelerated inference and Flask to run the UI page. This project was awarded first place in the Nvidia 9th Sky Hackathon competition.

The entire project workflow can be divided into the following steps:

1. Image Inference
2. Story Generation
3. Image Generation

![image-20231205130800981](./assets/image-20231205130800981.png)

## Prerequisites

### Prepare Model && Calibration Data

#### ONNX

Our project models are based on the mmpretrain pre-trained models from the mmlab algorithm library. We have carefully selected 25 classic backbone networks for the image classification task in this project. We also provide scripts for converting PyTorch (pt) models to ONNX models, including the recent work on EfficientVit. Additionally, we offer conversion scripts to export ONNX models in Dynamic Shape mode. 

We provide both preprocessed ONNX models using Polygraphy and the original exported ONNX model files, You can choose to download it from [Google Drive](https://drive.google.com/file/d/1T0-ZIWmZ7eQ7y6KvjXEUbsswfG1GvcNe/view?usp=drive_link) or [Hugging Face](https://huggingface.co/CtrlX/ModelReady-pretrain/tree/main).

Please place the downloaded ONNX file into the `models/onnx` directory.

#### Calibdata

Our calibration dataset consists of 510 images selected from the ImageNet 1K validation dataset. We also provide a [download link](https://drive.google.com/file/d/10QTdYG3SvPnC8xLRYmjBFWInl0qEgbza/view?usp=drive_link) for the calibration dataset. 

Please place the downloaded calibdata file into the `models/calibdata` directory.

### Prepare API

Before running this project, you need to prepare the [Nvidia NGC llama2-70b-steerlm API](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/llama2-70b-steerlm/api) and the [Nvidia NGC Stable Diffusion XL API](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/sdxl/api) and fill in their details in the `config.json` file. You can also fill in your Azure OpenAI API key in the config.json if you have one, but this is not mandatory.

```json
"sdxl": {
    "invoke_url": "" ,
    "fetch_url_format": "",
    "headers": {
        "Authorization": "",
        "Accept": "application/json"
    }
},
"llama2": {
    "invoke_url": "",
    "fetch_url_format": "",
    "headers": {
        "Authorization": "",
        "Accept": ""
    }
},
"azure_openai":{
    "api_key": "",
    "api_base": "",
    "deployment_name": "",
    "api_version": ""
}
```

### Setup Runtime Environment

We provide two methods for building the runtime environment for different hardware environments. One is deploying the environment using Nvidia Container on Windows or Linux, and the other is configuring the environment using pip on a Jetson Orin development board.

- Nvidia Jetson Xavier NX / Jetson nano
  - [Jetpack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461) (LT4 32.7.1 + TensorRT 8.2.1 + cuDNN 8.2.1 + CUDA 10.2)

> Note: If you are using the **x86 or Jetson Orin series** hardware platform, please refer to this project: https://github.com/1438802682/NarrativeCanvas

#### Jetson Xavier NX / Jetson nano

Before building the runtime environment on the Jetson platform, please upgrade the Jetson's JetPack environment version to 4.6.1, and then execute the following command:

```
pip3 install requirements.txt
```



## Run

We can then run the docker with the following command:

```py
gunicorn -b 127.0.0.1:3008 app:app
```

After you have completed the steps above, please visit http://127.0.0.1:3008/ to embark on your creative journey!



## Note

### Demonstration Sample

![image-20231205130723783](./assets/image-20231205130723783.png)

![image-20231205130713558](./assets/image-20231205130713558.png)

### UI Prototype

![](./assets/protograph.png)

### Project Architecture Diagram

![](./assets/architecture.png)

### Flowchart

![](./assets/flowsheet.png)



- If you encounter any issues or would like to obtain more technical details, please feel free to contact me at [1438802682@qq.com](mailto:1438802682@qq.com)

## References

- [Narrative Canvas](https://github.com/1438802682/NarrativeCanvas)
- [EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction](https://github.com/mit-han-lab/efficientvit)
- [MMPreTrain is an open source pre-training toolbox based on PyTorch](https://github.com/open-mmlab/mmpretrain/tree/main/mmpretrain)

