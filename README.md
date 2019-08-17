# Interpretable Feature Visualization of YOLOv3 Keras Model
A feature visualization tool for trained YOLOv3 models to gain better insight into how they perform classification and localization by generating saliency maps

## Motivation

Machine learning models are often perceived as “black-box models,” in which some type of data is fed into the model, which somehow processes the data, learning patterns and features, and then produces an output. The issue with the black-box model, however, is that we have no insights as to whether or not our models is optimizing the true objective. How can we truly trust a model and what it’s telling us about our data if we have no insight into how it is generating its output? 

When analyzing machine learning models, feature visualization provides a visual explanation as to which parts of the image are most influential in activating specific neurons which helped lead the model to its final prediction. 

Neural networks themselves are differentiable with respect to their inputs. To discover what parts of the input image helped the network produce certain predictions, through activating certain neurons, we can use backpropagation on a trained network with fixed weights learned during training. 


## How to Run Visualizations for YOLOv3 Model

### Training
1. Save weights & optimizers to an .h5 file during training

### Environment Setup
1. Clone the repository to your machine (within your YOLOv3 model directory)
2. Cd into ```visualize``` directory 
3. Create environment with the necessary dependencies: ```conda env create -f environment.yml```

### Visualizations Results
1. To run visualizations:
    - Load weights/optimizers into your model
    - Compile model 
    - Save model in your main executable .h5 file
    - Integrate summarize() or main_saliency() with the pre-trained model as desired:
        - NOTE: If using guided backpropagation with a custom loss function you've made:
            1. Instantiate your custom loss function
            2. Pass your custom loss function into your compile() function
            3. Before running guided backpropagation, enter: 
            ```
            get_custom_objects().update({"compute_loss": <your_loss_function_here>,'tf':tf})
            ```
## YOLOv3
[YOLOv3 ](https://pjreddie.com/media/files/papers/YOLOv3.pdf)is a real-time object detection algorithm, which uses a deep convolutional neural network to classify and locate objects. In short, the motivation behind the application of YOLOv3 lies in the algorithm’s high frame rate. Exhibiting the best performance compared to other object detection models such as SSD (Single Shot Multibox Detector) or Faster R-CNN, YOLOv3 yields the capability to most accurately detect small objects and fine details in images.

## Feature Visualization Methods
The methods implemented in this feature visualization tool derive from various techniques of backpropagating and calculating for gradients.

#### Plain "Vanilla" Gradients
This gradient-based saliency technique is based on the paper, [Deep Inside Convolutional Networks: Visualising
Image Classification Models and Saliency Maps ](https://arxiv.org/pdf/1312.6034.pdf). Given a prediction output, we map it back to the input pixel space to generate a saliency map by calculate the derviative of the class score with respect to the input image pixels. The value indicates which pixels are most sensitive to the class prediction score.

#### Integrated Gradients
Another gradient-based saliency technique, Integrated Gradients based on the paper, [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), calculates the integral of gradients along a uniform scale of varying pixel intesities to measure how influential a given pixels is with respect to our class prediction score. For this project, we start with a baseline image of all pixel values of 0 (black image) and uniformly scale this baseline image to the input image while uniformly scaling these pixel intensities, alpha. This straight-line path, as compared to most non-linear attribution methods, is used to measure how a single feature changes by calculating a Riemann sum to approximate the integral of these values. Note the default value of steps is set to 20, but the paper above notes that 20 to 300 steps are most appropriate for integral approximation.

#### Guided Backpropagation
With guided backpropagation, originating from [Striving For Simplicity: The All Convolutional Net](https://arxiv.org/pdf/1412.6806.pdf), we only backpropagate on positive gradients, thus indicating which input image pixels led to a positive class prediction. By backpropagating this way, we ignore irrelevant information, as in input pixels with corresponding negative gradients which led to a negative class prediction. 

#### Visual Backpropagation
This value-based method, as discussed in [VisualBackProp: efficient visualization of CNNs](https://arxiv.org/pdf/1611.05418.pdf), uses the concept of a deconvolutional network elaborated in Matthew D. Zeiler & Rob Fergus's paper, [Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf). The motivation behind the technique described in the VisualBackProp paper lies in the intuition that as we move further along a deep convolutional network, we lose important information. To counteract this loss of information, VisualBackProp considers the output of each convolutional layer, the feature maps, as points where the model holds the most relevant information. Following a single forward pass, we use a deconvolutional network along with VisualBackProp to map the feature maps of the last convolutional layer back to those of the first convolutional layers. With this method, we are able to preserve key information about the input, since we look at the feature maps of each convolutional layer throughout the network rather than mapping a single output layer back to the input pixel space. In terms of technical details, as a brief overview, we calculate the average of the feature maps following each LeakyReLU layer, then we use deconvolutional to upsample and scale the resulting product to the size of the feature maps of the previous layer. This continues until we reach the input pixel space. Note that the implementation discussed in the VisualBackProp paper has been modified for YOLOv3, in which I not only upsample via deconvolution, but I also downsample via AveragePool2D if the network backpropagates through Upsample2D layers. 

### Resources
Most feature visualization research within interpretability stems from the research discussed in[ Visualizing and Understanding Convolutional Networks](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf), by Matthew D. Zeiler and Rob Fergus.

Another great place to read about feature visualization is [this article from distill.pub](https://distill.pub/2017/feature-visualization/) 

