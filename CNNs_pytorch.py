"""# Convolutional Neural Networks in pytorch"""
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
import torch as torch
from classes import classes

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)])

# TODO you code here
# - create an object of type torchvision.datasets.OxfordIIITPet, download it
train_set = torchvision.datasets.OxfordIIITPet(
    root='OxfordPets',
    split='trainval',
    download=True,
    transform=transform)
# - torch.utils.data.DataLoader object
batch_size = 4
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


# - display some samples
def imshow(img):
    # un-normalize
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# get some random training images
print(train_set)
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

"""# 2. The Convolutional Neural Network"""
import torch.nn.functional as F


# TODO your code here: define a simple CNN model, pass a single example through the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120 * 10)
        self.fc2 = nn.Linear(120 * 10, 120)
        self.fc3 = nn.Linear(120, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


print(images.shape)
net = Net()
print("CNN from scracth for batch of 4 output:", net(images))

"""## Transfer learning"""

# TODO : your code here
# get a pretrained torchvision module, change the last layer,  pass a single example through the model
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(classes))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print("Transfer learning output for batch of 4:", model_ft(images))

"""# 3. Training the model

optimizer - the chosen optimizer. It holds the current state of the model and will update the parameters based on the computed gradients. Notice that in the constructor of the optimizer you need to pass the parameters of your model and the learning rate.
criterion - the chosen loss function.
"""
num_epochs = 50
loss_history=[]
for epoch in range(num_epochs):  # num_epochs is a hyperparameter that specifies when is the training process
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # iterate over the dataset, now we use data loaders
        # get a batch of data (inputs and their corresponding labels)
        inputs, labels = data
        # IMPORTANT! set the gradients of the tensors to 0.
        # by default torch accumulates the gradients on subsequent backward passes
        # if you omit this step, the gradient would be a combination of the old gradient,
        # which you have already used to update the parameters
        optimizer.zero_grad()
        # perform the forward pass through the network
        outputs = net(inputs)

        # apply the loss function to determine how your model performed on this batch
        loss = criterion(outputs, labels)
        loss_history.append(loss)
        # start the backprop process. it will compute the gradient of the loss with respect to the graph leaves
        loss.backward()

        # update the model parameters by calling the step function
        optimizer.step()
    print("Running loss for epoch ",epoch," :",running_loss)
exit()
"""

# TODO code to train your model

# TODO code to train your model

""""""Now let's examine the effect of the learning rate over the training process.

- First, create two plots: one in which you plot, for each epoch, 
the loss values on the training and the test data (two series on the same graph),
 and another one in which you plot, for each epoch, the accuracy values on the training and the test data.
- Experiment with different values for the learning rate.
- Then, experiment with a torch.optim.lr_scheduler to adjust the learning rate during the training process
 [doc](!https://pytorch.org/docs/stable/optim.html).

```
optimizer = SGD(model, lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(num_epochs):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    # apply the learning rate scheduler
    scheduler.step()
```

Plot the learning curves for all the training that you performed.
Fill in the table to compare the accuracy of your trained models.

| Model              | lr config            | accuracy  train| accuracy test |
| -----------        | -----------          | ------         | -----         |
| Model              | lr info              |   acc          |acc            |
| Model              | lr info              |   acc          |acc            |


You can work in teams and each team will train the model with a different setup.

# Using the GPU

``torch`` is designed to allow for computation both on CPU and on GPU.
If your system has a GPU and the required libraries configured for torch compatibility, the cell below will print information about its state.

If you are running your code on colab, you can enable GPU computation from: Runtime->Change Runtime type -> T4 GPU
"""

import torch

# decomment
# if torch.cuda.is_available():
#     !nvidia - smi
# else:
#     print("NO GPU ☹️")

"""Now we can start to use accelaration.
You now need to explictly specify on which device your tensors reside. You can
move all of the model's parameters `.to` a certain device (the GPU)
and also move the data on the same device there as well
before applying the model and calculating the loss.
"""
# decomment
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# loss_fn(model(x.to(device)), y.to(device))

"""#Useful references

- [1] [a "recipe" ](http://karpathy.github.io/2019/04/25/recipe/)  when you will start training artifcial neural networks;
- [2] [Defining a CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) in torch;
- [3] [Transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) in torch;
- [4] [model debugging](https://developers.google.com/machine-learning/testing-debugging/common/overview).

# <font color='red'> Optional </font>  
"""
