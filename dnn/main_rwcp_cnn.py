import torch
import time
import os

from model import CNN
from util import training_ann, testing_ann, get_train_loader, get_test_loader

# Data pre-processing
print('==> Preparing data..')
home_dir = os.getcwd()
train_data_dir = os.path.join(home_dir, 'data/RWCP_dataset_original.mat')
test_data_dir_original = os.path.join(home_dir, 'data/RWCP_dataset_original.mat')
test_data_dir_masked60 = os.path.join(home_dir, 'data/RWCP_dataset_masked60.mat')
test_data_dir_masked70 = os.path.join(home_dir, 'data/RWCP_dataset_masked70.mat')
test_data_dir_masked80 = os.path.join(home_dir, 'data/RWCP_dataset_masked80.mat')
test_data_dir_masked90 = os.path.join(home_dir, 'data/RWCP_dataset_masked90.mat')
test_data_dir_reconstruct60 = os.path.join(home_dir, 'data/RWCP_dataset_reconstruct_algorithm_ours_60.mat')
test_data_dir_reconstruct70 = os.path.join(home_dir, 'data/RWCP_dataset_reconstruct_algorithm_ours_70.mat')
test_data_dir_reconstruct80 = os.path.join(home_dir, 'data/RWCP_dataset_reconstruct_algorithm_ours_80.mat')
test_data_dir_reconstruct90 = os.path.join(home_dir, 'data/RWCP_dataset_reconstruct_algorithm_ours_90.mat')

train_loader = get_train_loader(train_data_dir)
test_loader = get_test_loader(test_data_dir_original)
test_loader_mask90 = get_test_loader(test_data_dir_masked90)
test_loader_reconstruct90 = get_test_loader(test_data_dir_reconstruct90)
test_loader_mask80 = get_test_loader(test_data_dir_masked80)
test_loader_reconstruct80 = get_test_loader(test_data_dir_reconstruct80)
test_loader_mask70 = get_test_loader(test_data_dir_masked70)
test_loader_reconstruct70 = get_test_loader(test_data_dir_reconstruct70)
test_loader_mask60 = get_test_loader(test_data_dir_masked60)
test_loader_reconstruct60 = get_test_loader(test_data_dir_reconstruct60)

if __name__ == '__main__':
    # CUDA configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        device = 'cuda'
        print('GPU is available')
    else:
        device = 'cpu'
        print('GPU is not available')

    # Parameters
    num_epochs = 50
    global best_acc
    best_acc = 0

    # Models and training configuration
    model = CNN()
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        since = time.time()

        # Training Stage
        model, acc_train, loss_train = training_ann(model, train_loader, optimizer, criterion, device)

        # Testing Stage
        acc_test, loss_test = testing_ann(model, test_loader, criterion, device)
        acc_test_mask90, loss_test_mask90 = testing_ann(model, test_loader_mask90, criterion, device)
        acc_test_reconstruct90, loss_test_reconstruct90 = testing_ann(model, test_loader_reconstruct90, criterion, device)
        acc_test_mask80, loss_test_mask80 = testing_ann(model, test_loader_mask80, criterion, device)
        acc_test_reconstruct80, loss_test_reconstruct80 = testing_ann(model, test_loader_reconstruct80, criterion, device)
        acc_test_mask70, loss_test_mask70 = testing_ann(model, test_loader_mask70, criterion, device)
        acc_test_reconstruct70, loss_test_reconstruct70 = testing_ann(model, test_loader_reconstruct70, criterion, device)
        acc_test_mask60, loss_test_mask60 = testing_ann(model, test_loader_mask60, criterion, device)
        acc_test_reconstruct60, loss_test_reconstruct60 = testing_ann(model, test_loader_reconstruct60, criterion, device)

        # Training Record
        time_elapsed = time.time() - since
        print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
        print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
        print('Test Accuracy Clean: {:4f}'.format(acc_test))
        print('Test Accuracy after Mask (90%): {:4f}'.format(acc_test_mask90))
        print('Test Accuracy after Reconstruction(90%): {:4f}'.format(acc_test_reconstruct90))
        print('Test Accuracy after Mask (80%): {:4f}'.format(acc_test_mask80))
        print('Test Accuracy after Reconstruction(80%): {:4f}'.format(acc_test_reconstruct80))
        print('Test Accuracy after Mask (70%): {:4f}'.format(acc_test_mask70))
        print('Test Accuracy after Reconstruction(70%): {:4f}'.format(acc_test_reconstruct70))
        print('Test Accuracy after Mask (60%): {:4f}'.format(acc_test_mask60))
        print('Test Accuracy after Reconstruction(60%): {:4f}'.format(acc_test_reconstruct60))

