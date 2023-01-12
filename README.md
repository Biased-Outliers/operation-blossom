# Operation Blossom
You are taking a casual, brisk morning walk when suddenly you saw a beautiful flower that almost made you trip over yourself trying to take a second look. You are fascinated with the flower but you do not know what type it is. This is when you Blossom! 

Blossom is a computer vision app where you can upload a picture of a flower and it will tell you what flower it is and few fascinating facts about it. Also for those computer vision nerds (like ourselves), we show results of two popular computer vision architectures, Convolutional Neural Networks and Vision Transformers, each showing their respective results. 

The minimum viable product for Blossom is it can classify a flower into 5 categories: daisy, sunflower, dandelion, rose, tulip.

## Blossom's Architecture
As of now, Blossom utilizes two different computer vision architectures: Convolutional Neural Networks (ConvNets) and Vision Transformers (ViT). For the ConvNets, we transfered the ResNet50 model using the Tensorflow API and retrained it to fit our flower classification. For the ViT, we transfered the [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) model using the HuggingFace API and retrained it to also fit the flower classification. The ViT trained model can be found here: [Blossom ViT](https://huggingface.co/taraqur/blossom-vit). The results for both architecture are displayed in the Streamlit app for those who are curious.

## TechStack
* FrontEnd: Streamlit
* Language: Python
* Frameworks
  * Tensorflow for ConvNet: [ResNet50 using Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
  * HuggingFace for ViT: [Blossom ViT](https://huggingface.co/taraqur/blossom-vit)

## Resources
* ResNet50
  * Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)
  * [Tensorflow API](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
* Vision Transformers
  * Paper: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)
  * [HuggingFace Implementation](https://theaisummer.com/hugging-face-vit/)
  * [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)

## To Do's
* Add details of the predicted flower
* Add more images for training
* Expand the classification to more flower species
* Add ability to upload multiple pictures and predict them all
* Implement GRAD CAMs
