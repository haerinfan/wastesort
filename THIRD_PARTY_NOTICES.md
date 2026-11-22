# Third-party attribution

## TrashNet

Dataset creators: Gary Thung and Mindy Yang.

Source: https://github.com/garythung/trashnet

The creators' Hugging Face dataset card identifies the dataset license as MIT. The following notice is retained from the original repository:

> MIT License
>
> Copyright (c) 2017 Gary Thung
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## MobileNetV2 and ImageNet

Architecture: Sandler, Howard, Zhu, Zhmoginov and Chen (2018), *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, https://doi.org/10.1109/CVPR.2018.00474.

Implementation and pretrained parameters: torchvision `MobileNet_V2_Weights.IMAGENET1K_V2`, https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.mobilenet_v2.html.

Torchvision source license: https://github.com/pytorch/vision/blob/v0.22.1/LICENSE.
Pretrained ImageNet parameters are a dependency of the transfer-learning checkpoint. Their training data and applicable terms are not replaced by this project's notices. This project makes no claim that it collected or owns ImageNet or TrashNet.

## Project code

No additional open-source license is granted for the group's original code in this initial setup. Public visibility permits viewing the work but is not a substitute for a group-approved license. Third-party licenses continue to apply to their respective components.

## Primary set (synthetic) tools

The primary-set images are generated with **SDXL-Turbo** by Stability AI (https://huggingface.co/stabilityai/sdxl-turbo, revision 71153311d3dbb46851df1931d3ca6e939de83304). It is used under the Stability AI Non-Commercial Research Community License, which the model card describes as allowing non-commercial and research use. The generated images are used only in this non-commercial course study.

Quality control uses **CLIP ViT-B/32** by OpenAI (https://huggingface.co/openai/clip-vit-base-patch32, revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268), released by OpenAI under the MIT license in https://github.com/openai/CLIP.
