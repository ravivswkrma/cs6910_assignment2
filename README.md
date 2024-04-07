# cs6910_assignment2

About the Dataset
iNaturalist 12k is a subset of the larger iNaturalist dataset that contains around 12,000 observations of over 4,300 plant and animal species.

Libraries used
Pytorch and torchvision

Part A
Create a compact CNN model comprising five convolutional layers, where each layer is succeeded by an activation function and a max-pooling layer. Following these five conv-activation-maxpool blocks, include a single dense layer and an output layer with ten neurons, corresponding to the ten classes in the dataset. Ensure that the input layer is compatible with the images in the iNaturalist dataset. The code should offer flexibility to adjust parameters such as the number and size of filters, activation functions for both convolutional and dense layers, and the number of neurons in the dense layer.
Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. The sweep configuration and default configurations of hyperparameters are specficied as follows:

configure sweep parameters:-
sweep_config={
"method": "bayes", "metric": { "name": "val acc", "goal": "maximize"
}, "parameters": { "max_epochs": { "values": [5,7,9] }, "num_filter":{ "values":[[64,64,64,64,64],[32,64,128,256,512],[32,32,32,32,32],[512,256,128,64,32]] }, "filter_size":{ "values":[[3,3,3,3,3],[5,5,5,5,5],[11,9,7,5,3]]
}, "cnn_act_fun":{ "values":['relu','gelu','mish','silu'] }, "data_aug":{ "values":[True,False] }, "batch_norm": { "values": [True,False] }, "dense_act_fun":{ "values":['relu','gelu','mish','silu'] }, "dropout":{ "values":[0.1,0.2,0.3] }, "dense_size":{ "values":[128,256,512] }, "mystride":{ "values":[2,3,5] } } }


Part B
In most DL applications, instead of training a model from scratch, we would use a model pre-trained on a similar/related task/dataset.I have used Resnet50model.
I utilized the ResNet50 model for this task.

a. Initially, I resized all images in our dataset to match the standard ImageNet image size of (224x224). This was achieved using APIs provided by various libraries, which implement different interpolation techniques for image resizing.

b. As the ImageNet dataset comprises 1000 classes, the last layer of every pre-trained model typically contains 1000 nodes. However, to train models with a different number of classes, I removed the last layer of the pre-trained ResNet50 model. Then, I added a dense layer of the desired size, such as 10, with a softmax activation function.

To achieve the desired output, I devised a strategy to retrain specific layers of the model while freezing others. Initially, I modified the last layer to match the number of classes in our dataset, freezing all layers except the last one. Subsequently, I initiated the retraining process for the fully connected layer.

The optimal approach to tackle this problem involves freezing certain layers, particularly the initial ones in the CNN architecture, while fine-tuning other layers. This ensures that only the neurons responsible for learning complex features are modified during backpropagation, while preserving those learning fundamental features.

In leveraging pre-trained models like ResNet50, I adopted different strategies for freezing specific stages during training. One method involved freezing all layers of the base convolutional layer except the last one. Alternatively, I froze the first k layers and trained the remaining ones from scratch, or froze the first k layers and initialized the remaining layers with pre-trained weights.

Moreover, I extended the model by adding more layers at the end. Since the initial layers of the model are trained on a general dataset, they capture broad features. By appending additional layers at the end, the model can learn more specific features.
