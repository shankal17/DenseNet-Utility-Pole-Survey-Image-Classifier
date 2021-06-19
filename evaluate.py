"""This module contains the functions used to evaluate a trained image classifier

Functions
---------
evaluate(model_path, classes, data_dir='/.data', log_name=None, batch_size=16)
    Calculates and logs the accuracy of the trained model

"""

import sys
import torch
import logging

from tqdm import tqdm
from torchvision import transforms, datasets
from models.Custom_DenseNet121 import CustomDenseNet121

def evaluate(model_path, classes, data_dir='/.data', log_name=None, batch_size=16):
    """Calculates and logs the accuracy of the trained model

    Parameters
    ----------
    model_path : string
        Path to model checkpoint to evaluate
    classes : list or tuple <string>
        iterable of class names
    data_dir : string, optional
        Directory path to split dataset
    log_name : string, optional
        File name or path in which to save the results
    batch_size : int, optional
        Batch size to use

    Raises
    ------
    AssertionError
        If length of the iterable of class names does not match the data given
    """

    # Define test transforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    num_classes = len(classes)
    assert num_classes == len(test_data.classes), "'classes' argument is of wrong size!"

    # Load CNN
    classifier = CustomDenseNet121(num_classes)
    classifier.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    accuracy = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad(): # Loop through testset
        for data in tqdm(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            logps = classifier(images)

            # Calculate overall accuracy and accuracy per class
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += equals[i].item()
                class_total[label] += 1

    # Log result to file
    if not log_name:
        log_name = "output.log"
    model_stats_logger = logging.getLogger()
    model_stats_logger.setLevel(logging.DEBUG)

    output_file_handler = logging.FileHandler(log_name)
    stdout_handler = logging.StreamHandler(sys.stdout)

    model_stats_logger.addHandler(output_file_handler)
    model_stats_logger.addHandler(stdout_handler)
    model_stats_logger.debug(f"Test accuracy: {100 * accuracy/len(testloader):.3f}")
    for i in range(num_classes):
        model_stats_logger.debug('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    data_dir = 'data/processed/train_val_split'
    classes = ('Birthmark', 'Grounded', 'Height Shot', 'Midspan', 'Pole Tag', 'Upshot')
    model_path = 'checkpoint_60.pth'
    
    evaluate(model_path, classes, data_dir=data_dir)