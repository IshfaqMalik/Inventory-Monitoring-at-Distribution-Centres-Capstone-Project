#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets 
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, random_split
import sys
import argparse
import logging
import os
import copy
import argparse
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, hook ):
    
    '''
    Evaluates the model on the test dataset to get test accuracy and loss.

    Args:
        model (torch.nn.Module): The neural network model to test.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        

    Returns:
        tuple: A tuple containing (average_loss, accuracy) on the test set.
    '''
    test_loss = 0
    correct=0
    total=0

    model.to('cpu')
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    logger.info("Starting model evaluation...")

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to('cpu'), target.to('cpu')
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()*data.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        average_loss = test_loss/total 
        accuracy = 100.0 * correct/total 

        logger.info(f"Test Average Loss: {average_loss}")
        logger.info(f"Test Accuracy: {accuracy:.2f}")

        return average_loss, accuracy
        
            
def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs, hook):
    '''
    Trains a PyTorch model, including training and validation phases,
    loss and accuracy tracking, best model saving, and basic early stopping.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function (e.g., nn.CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., optim.Adam, optim.SGD).
        device (torch.device): The device to run training on (e.g., 'cuda' or 'cpu').
        epochs (int): The number of training epochs.

    Returns:
        torch.nn.Module: The trained model with the best validation loss weights.
    '''
    best_loss = float('inf')
    dataloader = {"train":train_loader, "valid":valid_loader}
    best_model_wts = copy.deepcopy(model.state_dict())
    loss_counter = 0
    logger.info(f"Starting training for {epochs} epochs on device: {device}")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        for phase in ["train", "valid"]:
            if phase =="train":
                hook.set_mode(smd.modes.TRAIN)
                model.train()
            else: 
                hook.set_mode(smd.modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for data, target in dataloader[phase]:
                data, target = data.to(device), target.to(device)

                with torch.set_grad_enabled(phase=="train"):
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    _,preds = torch.max(outputs,1)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss  += loss.item()*data.size(0)
                running_corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss/len(dataloader[phase])
            epoch_acc = running_corrects/len(dataloader[phase])
            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid":
                if epoch_loss <best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    loss_counter = 0
                    logger.info(f"Validation loss improved. Saving model weights.")
            else:
                logger.info(f"Validation loss did not improve.")
                loss_counter+=1

    logger.info(f"Training complete. Best val loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model
            
            


    
def net(num_classes, device):
    '''
        Initializes a pre-trained EfficientNetB3 model, freezes its
    feature extraction layers, and replaces the classifier head
    for a new number of classes.

    Args:
        num_classes (int): The number of output classes for the new classification head.
        device (torch.device or str): The device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        torch.nn.Module: The initialized model ready for fine-tuning.
    '''
    model = models.efficientnet_b3(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Linear
                                     (num_features, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(512, num_classes, bias=True)
                                    )
    model.to(device)
    return model
        
        
class TransformedDataset(torch.utils.data.Dataset): 
    def __init__(self, subset, transform): 
        self.subset = subset 
        self.transform = transform 
        
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        x= self.transform(x)
        return x, y
        


def create_data_loaders(data_dir, batch_size):
    
    '''
    Creates training, validation, and test data loaders from a given directory.
    Assumes data is organized as: data_dir/class_name/image.jpg
    Applies appropriate transformations for training (augmentation) and
    validation/testing (resizing and normalization).

    Args:
        data_dir (str): The path to the root directory of the dataset.
        batch_size (int): The batch size for the data loaders.

    Returns:
        train_loader, valid_loader, test_loader
    '''
    train_transforms = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225]),
    ]) 
    test_transforms = transforms.Compose([ 
        transforms.Resize(256),
        transforms.CenterCrop(224),         
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225]),
    ]) 
    dataset = datasets.ImageFolder(data_dir) 

    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])

    train_data = TransformedDataset(train_set, transform = train_transforms) 
    test_data = TransformedDataset(test_set, transform = test_transforms)
    valid_data = TransformedDataset(valid_set, transform = test_transforms)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True) 
    test_loader = DataLoader(test_data, batch_size = batch_size) 
    valid_loader = DataLoader(valid_data, batch_size = batch_size) 

    return train_loader, test_loader, valid_loader
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')

    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    model=net(args.num_classes, args.device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    logger.info("Registering SMDebug hooks...")
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info("Starting Model Training")
    model=train(model, train_loader,valid_loader, loss_criterion, optimizer, args.device, args.epochs, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing Model")
    test(model, test_loader, loss_criterion,hook)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    save_file = os.path.join(args.save_path, "model.pth")
    torch.save(model.cpu().state_dict(), save_file)
    logger.info(f"Model saved to: {save_file}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--data_dir" , type=str,  default = os.environ.get("SM_CHANNEL_TRAINING", "data"))
    parser.add_argument("--save_path", type = str, default = os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default = 0.001)
    parser.add_argument("--num_classes", type=int, default =5)
    parser.add_argument("--epochs", type = int, default =50)
    parser.add_argument("--device", type = str, default = "cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    
    main(args)
