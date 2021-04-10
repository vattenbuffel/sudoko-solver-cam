import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import Compose
import PIL





class Digit_recognizer(nn.Module):
    def __init__(self, transforms, outputs=None, L2_penalty=0, learning_rate=0.01):
        super().__init__()
        
        
        self.norm0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(40)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        self.dropout2 = nn.Dropout(p=0.2, inplace=False)
        self.linear_regression_vector_length = 960
        self.nn1 = nn.Linear(self.linear_regression_vector_length, 256)
        self.nn2 = nn.Linear(256, 9)
        
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), learning_rate,  weight_decay = L2_penalty) # weight_decay is the alpha in weight penalty regularization
            
        # Move this to device
        self.to(device)

        # Set up transforms
        self.transforms = transforms
        
    def forward(self, x): 
        x = self.norm0(x)
        x = self.conv1(x)
        x = self.dropout1(x)
        x = F.relu(x)

        
        x = self.norm1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = F.relu(x)

        x = self.norm2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = torch.reshape(x, (-1, self.linear_regression_vector_length))
        x = self.nn1(x)
        x = F.relu(x)

        x = self.nn2(x)
        x = F.log_softmax(x)
            
        return x
            
    def backward(self, predicted, truth):
        loss = self.loss_fn(predicted, truth)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def train_(self, train_loader, val_loader, epochs, dict_of_loss_and_accuracy = None, save_best=True):
        train_n = len(train_loader.dataset)
        lowest_val_loss = 1e10
        
        for i in range(epochs):
            losses = []
            n_correct = 0
            self.train()
            for x,y in train_loader:
                pred = self.forward(x.to(device))
                loss = self.backward(pred, y.to(device))

                losses.append(loss.item())
                n_correct += (pred.argmax(dim=1) == y.to(device)).sum().item()


            train_accuracy = n_correct / train_n
            train_avg_loss = sum(losses) / len(losses)
            self.eval()
            val_accuracy, val_avg_loss = self.eval_(val_loader)
            
            if not (dict_of_loss_and_accuracy is None):
                dict_of_loss_and_accuracy['training_loss'].append(train_avg_loss)
                dict_of_loss_and_accuracy['validation_loss'].append(val_avg_loss)
                dict_of_loss_and_accuracy['training_accuracy'].append(train_accuracy)
                dict_of_loss_and_accuracy['validation_accuracy'].append(val_accuracy)
                if save_best and lowest_val_loss > val_avg_loss:
                    torch.save(self.state_dict(), "network")

            print("After epoch: {}\tVal_avg_loss: {:.3f}\tTrain_avg_loss: {:.3f}\tVal_accuracy: {:.3f}\tTrain_accuracy: {:.3f}".format(i, val_avg_loss, train_avg_loss, val_accuracy, train_accuracy))

    def eval_(self, val_data_loader): 
        losses = []
        n_correct = 0
        with torch.no_grad():
            for x, y in val_data_loader:
                pred = self(x.to(device)) 
                loss = self.loss_fn(pred, y.to(device))
                losses.append(loss.item())

                n_correct += torch.sum(pred.argmax(dim=1) == y.to(device)).item()
            val_accuracy = n_correct/len(val_data_loader.dataset)
            val_avg_loss = sum(losses)/len(losses)    

        return val_accuracy, val_avg_loss

    def predict_on_image(self, img):
        """
        Returns an prediction based on img. 10 is blank, the rest shows their actual value
        """
        img_pil = PIL.Image.fromarray(img)
        img_processed = self.transforms(img_pil)
        img_processed = img_processed.reshape(1,img_processed.shape[0], img_processed.shape[1], img_processed.shape[2])
        with torch.no_grad():
            prediction = self(img_processed.to(device))
        return torch.argmax(prediction[0]).item()+1

    def compare_transformss(self, transformations, index):
        """Visually compare transformations side by side.
        Takes a list of ImageFolder datasets with different compositions of transformations.
        It then display the `index`th image of the dataset for each transformed dataset in the list.
        
        Example usage:
            compare_transforms([dataset_with_transform_1, dataset_with_transform_2], 0)
        
        Args:
            transformations (list(ImageFolder)): list of ImageFolder instances with different transformations
            index (int): Index of the sample in the ImageFolder you wish to compare.
        """
        
        # Here we combine two neat functions from basic python to validate the input to the function:
        # - `all` takes an iterable (something we can loop over, like a list) of booleans
        #    and returns True if every element is True, otherwise it returns False.
        # - `isinstance` checks whether a variable is an instance of a particular type (class)
        if not all(isinstance(transf, ImageFolder) for transf in transformations):
            raise TypeError("All elements in the `transformations` list need to be of type ImageFolder")
            
        num_transformations = len(transformations)
        fig, axes = plt.subplots(1, num_transformations)
        
        # This is just a hack to make sure that `axes` is a list of the same length as `transformations`.
        # If we only have one element in the list, `plt.subplots` will not create a list of a single axis
        # but rather just an axis without a list.
        if num_transformations == 1:
            axes = [axes]
            
        for counter, (axis, transf) in enumerate(zip(axes, transformations)):
            axis.set_title("transf: {}".format(counter))
            image_tensor = transf[index][0]
            self.display_image(axis, image_tensor)

        plt.show()

    def display_image(self, axis, image_tensor):
        """Display a tensor as image
        
        Example usage:
            _, axis = plt.subplots()
            some_random_index = 453
            image_tensor, _ = train_dataset[some_random_index]
            display_image(axis, image_tensor)
        
        Args:
            axis (pyplot axis)
            image_tensor (torch.Tensor): tensor with shape (num_channels=3, width, heigth)
        """
        
        # See hint above
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("The `display_image` function expects a `torch.Tensor` " +
                            "use the `ToTensor` transformation to convert the images to tensors.")
            
        # The imshow commands expects a `numpy array` with shape (3, width, height)
        # We rearrange the dimensions with `permute` and then convert it to `numpy`
        image_data = image_tensor.permute(1, 2, 0).numpy()
        height, width, _ = image_data.shape
        axis.imshow(image_data)
        axis.set_xlim(0, width)
        # By convention when working with images, the origin is at the top left corner.
        # Therefore, we switch the order of the y limits.
        axis.set_ylim(height, 0)
        


def load_model(path="network"):
    model = Digit_recognizer(Compose(list_of_transforms))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    return model


image_width = 30
list_of_transforms = [transforms.Resize((image_width,image_width)),  
                    # transforms.CenterCrop(23),
                    transforms.Grayscale(1), 
                    transforms.ToTensor()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Digit-recogniser uses device:", device)

if __name__ == '__main__':
    train_path = "./img/datasets/combined/training-set/"
    val_path = "./img/datasets/combined/validation-set/"

    batch_size = 32
    num_workers = 1
    learning_rate = 0.0005
    n_epochs = 100
    
    digit_recognizer = Digit_recognizer(Compose(list_of_transforms), learning_rate = learning_rate)

    train_folder = ImageFolder(train_path, transform=Compose(list_of_transforms))
    train_loader = DataLoader(dataset=train_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_folder = ImageFolder(val_path, transform=Compose(list_of_transforms))
    val_loader = DataLoader(dataset=val_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Showing some train images
    # digit_recognizer.compare_transformss([train_folder, train_folder], 0)

    
    # with torch.cuda.device(0):
    dict_of_loss_and_accuracy = {'training_loss':[], 'validation_loss':[], 'training_accuracy':[], 'validation_accuracy':[]}
    digit_recognizer.train_(train_loader, val_loader, n_epochs, dict_of_loss_and_accuracy=dict_of_loss_and_accuracy)

    # torch.save(digit_recognizer.state_dict(), "network")






