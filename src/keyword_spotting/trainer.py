# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import time
import random

import torch
import numpy as np
from torchinfo import summary

import matplotlib.pyplot as plt


def train_kws(model, fe, augments, label_based_augments, train_loader, validation_loader, optimizer, criterion, device,
    tensorboard_writer, epochs, batch_size, model_path, lr_patience=20, patience=60, net_class='rnn', one_hot=True, freq_mix_style=None):
    """
    Trainer for KWS.
    """

    # Metrics
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    epoch_times = []

    # Early stopping
    best_loss_validation = np.inf
    best_epoch = 0
    lr_patience_counter = lr_patience
    patience_counter = patience
    epochs_to_smoth_val_loss = 3

    # Get trainable parameters:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ' + str(trainable_params))

    # For present intermediate information
    n_intermediate_steps = int(len(train_loader)/3) if int(len(train_loader)/3) > 1 else 1

    # Move again the model to device
    model = model.to(device)
    
    # Start training loop
    print('Starting trainig...')
    for epoch in range(1, epochs+1):
        start_time = time.time()
        
        train_loss = 0.0
        train_accuracy = 0.0
        counter = 0

        model.train()
        for _, noisy, _, target in train_loader:
            counter += 1

            # Initilize gradients
            model.zero_grad()

            # Move data to working device
            x = noisy.to(device)

            # Data augmentation
            x = augments(x)

            # Label based data augmentation
            x, target = label_based_augments(x, target)

            # Feature extractor
            x = fe(x)

            # Frequency mix style: groups samples by (dominant) class
            if(freq_mix_style):
                x = freq_mix_style(x, target.argmax(1) if target.dim() > 1 else target)

            # Model forward: rnn or cnn
            if(net_class == 'rnn'):
                h = model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = model(x.to(device).float())

            # Backward and optimization
            loss = criterion(out, target.to(device))
            loss.backward()
            optimizer.step()

            # Store metrics
            train_accuracy += right_predictions(out, target)
            train_loss += loss.item()

            # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Average Loss: {} | Accuracy for Step: {}".format(
                    epoch,
                    counter,
                    len(train_loader),
                    round(train_loss/counter, 4),
                    round(train_accuracy/(counter * batch_size), 4)
                    ))

        # Validate model
        validation_loss = 0.0
        validation_accuracy = 0.0

        with torch.no_grad():
            model.eval()
            for _, noisy, _, target in validation_loader:
                
                # Move data to working device
                x = noisy.to(device)

                # Augments
                # FIXME: This does not make sense while validating
                # The idea is to pick the model that best matches the real usage conditions
                x = augments(x)

                # Feature extraction
                x = fe(x)

                # Model forward
                if(net_class == 'rnn'):
                    h = model.init_hidden(x.shape[0], device) # Memory reset
                    h = h.data
                    out, h = model(x.to(device).float(), h)
                elif(net_class == 'cnn'):
                    out = model(x.to(device).float())
                
                # Store metrics: loss and accuracy
                if(one_hot):
                    loss = criterion(out, target.to(device).long())
                else:
                    loss = criterion(out, target.to(device).float().view_as(out))
                validation_accuracy += right_predictions(out, target)
                validation_loss += loss.item()

        # Calculate average losses
        train_loss = train_loss/len(train_loader)
        validation_loss = validation_loss/len(validation_loader)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        # Calculate accuracies
        train_accuracy = train_accuracy/len(train_loader.sampler)
        validation_accuracy = validation_accuracy/len(validation_loader.sampler)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        smothed_val_loss = np.mean(validation_losses[-epochs_to_smoth_val_loss:])
        
        # Print epoch information
        current_time = time.time()
        print("")
        print("Epoch {}/{} Done.".format(epoch, epochs))
        print("\t Tain Loss: {:.6f}  Validation Loss: {:.6f}".format(train_loss, validation_loss))
        print("\t Train accuracy: {:.6f}    Validation accuracy: {:.6f}".format(train_accuracy, validation_accuracy))
        print("\t Best Validation Loss: {:.6f}    Smothed Validation Loss: {:.6f}".format(best_loss_validation, smothed_val_loss))
        print("\t Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
        print("")
        epoch_times.append(current_time-start_time)

        # Early stopping
        if(best_loss_validation <= smothed_val_loss):
            patience_counter += 1
            lr_patience_counter += 1

            print('Validation loss did not improve {:.3f} vs {:.3f}. Patience {}/{}.'.format(
                smothed_val_loss,
                best_loss_validation,
                patience_counter,
                patience
                ))
            print("")
            
            if(lr_patience_counter == lr_patience):
                lr_patience_counter = 0
                optimizer, old_lr, new_lr = decrease_lr(optimizer, factor=10)
                print('Updating lr, previous value: {0}, new_value: {1}'.format(old_lr, new_lr))
                print("")
            
            if(patience_counter == patience):
                print('Breaking train loop: out of patience')
                print("Saving last model....\n")
                model.save_entire_model(model_path + '_last')
                break

        else:
            # Reinitialize patience counter and save model
            best_epoch = epoch
            patience_counter = 0
            lr_patience_counter = 0
            best_loss_validation = smothed_val_loss
            model.save_entire_model(model_path)
            model.save(model_path)

        # Tensorboard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
        tensorboard_writer.add_scalar("train_accuracy", train_accuracy, epoch)
        tensorboard_writer.add_scalar("val_loss", validation_loss, epoch)
        tensorboard_writer.add_scalar("smothed_val_loss", smothed_val_loss, epoch)
        tensorboard_writer.add_scalar("val_accuracy", validation_accuracy, epoch)
        tensorboard_writer.add_scalar("best_val_loss", best_loss_validation, epoch)
        tensorboard_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        tensorboard_writer.add_scalar("lr_patience", lr_patience_counter, epoch)
        tensorboard_writer.add_scalar("patience", patience_counter, epoch)
        tensorboard_writer.add_scalar("epoch_time", epoch_times[-1], epoch)

    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    metrics = {}
    metrics['training_time_s'] = sum(epoch_times)
    metrics['number_of_epochs'] = len(epoch_times)
    metrics['train_loss'] = train_losses
    metrics['validation_loss'] = validation_losses
    metrics['train_accuracy'] = train_accuracies
    metrics['validation_accuracy'] = validation_accuracies

    # Tensorboard
    tensorboard_writer.add_scalar("training_time", sum(epoch_times))
    tensorboard_writer.add_scalar("best_epoch", best_epoch)

    return model, metrics


def test_kws(wuw_model, fe, test_loader, criterion, device, batch_size,
    net_class='rnn', one_hot=False, se_model=None):
    
    # Metrics initialization
    test_loss = 0.0
    test_accuracy = 0.0

    labels = []
    predicted_scores = []
    predicted_labels = []
    ids = []

    # Move models to the same device
    wuw_model = wuw_model.to(device)
    if(se_model):
        se_model = se_model.to(device)

    # For present intermediate information
    n_intermediate_steps = int(len(test_loader)/3) if int(len(test_loader)/3) > 1 else 1

    counter = 0

    with torch.no_grad():

        wuw_model.eval()
        if(se_model):
            se_model.eval()

        if len(test_loader.dataset) < batch_size:
            batch_size = len(test_loader.dataset)
        
        for ID, noisy, _, target in test_loader:
            counter += 1
            
            # Move data to device
            x = noisy.to(device).float()

            # Speech Enhancement
            if(se_model):
                x = se_model(x)

            # Feature Extraction
            x = fe(x)

            # WuW forward
            if(net_class == 'rnn'):
                h = wuw_model.init_hidden(x.shape[0], device) # Memory reset
                h = h.data
                out, h = wuw_model(x.to(device).float(), h)
            elif(net_class == 'cnn'):
                out = wuw_model(x.to(device).float())

            # Store metrics: loss and accuracy
            if(one_hot):
                loss = criterion(out, target.to(device).long())
            else:
                loss = criterion(out, target.to(device).float().view_as(out))
            test_accuracy += right_predictions(out, target)
            test_loss += loss.item()

            # Store targets and predictions
            #labels += target.squeeze(1).tolist() #old
            #ids += ID.squeeze(1).tolist() #old

            # NOTE: This is a workaround to parse the ID tensor to list
            labels += target.tolist()
            try:
                ids += ID.tolist()
            except:
                ids += list(ID)

            if (one_hot):
                predicted_scores += torch.nn.functional.softmax(out.squeeze(1), 1).tolist()
                predicted_labels += torch.nn.functional.softmax(out, 1).data.max(1, keepdim=True)[1].squeeze(1).tolist()
            else:
                predicted_scores += out.squeeze().tolist()
                predicted_labels += (out.squeeze() > 0.5).long().tolist()

            # Present intermediate results
            if (counter%n_intermediate_steps == 0):
                print("Epoch {}......Step: {}/{}....... Average Loss, Accuracy for Step: {}, {}".format(
                    1,
                    counter,
                    len(test_loader),
                    round(test_loss/counter, 4),
                    round(test_accuracy/(counter * batch_size), 4)
                    ))

        # Calculate average losses
        test_loss = test_loss/len(test_loader)

        # Calculate accuracies
        test_accuracy = test_accuracy/len(test_loader.sampler)

        metrics = {}
        metrics['test_loss'] = test_loss
        metrics['test_accuracy'] = test_accuracy

    return ids, labels, predicted_scores, predicted_labels, metrics


def right_predictions(out, target, threshold=0.5):
    """
    From a given set of labes and predictions 
    it uses a threshols to determine if the prediction
    is right
    """
    counter = 0

    # For classifiers with two outputs
    if(out.shape[1] > 1):
        out = torch.nn.functional.softmax(out, 1)
        predictions_index = out.data.cpu().max(1, keepdim=True)[1]
        if len(target.shape) == 1:
            target = target.data.cpu().view_as(predictions_index)
        else:
            # FIXME: This is a workaround to work with mixup
            # Calculate target index based on probabilities
            target = target.data.cpu().max(1, keepdim=True)[1]
        counter = predictions_index.eq(target).sum().item()   
    # For classifiers with single output
    else:
        out = out.view_as(target).data.cpu() > threshold
        counter = (target == out).sum().item()
    return counter


def decrease_lr(optimizer, factor=10):
    old_lr = optimizer.param_groups[0]['lr']
    new_lr = old_lr / factor
    optimizer.param_groups[0]['lr'] = new_lr
    return optimizer, old_lr, new_lr


def set_seed(seed):
    """
    Fix seed of torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)


def get_forward_time(model, device, shape, net_class):
    """
    Returns CPU forward time in seconds
    """
    model.to('cpu')
    dummy_data = torch.rand(shape)
    if (net_class == 'rnn'):
        hidden_size =  model.init_hidden(1, torch.device('cpu')).data.numpy().shape
        h = torch.randn(hidden_size)
        start = time.time()
        _, _ = model(dummy_data, h)
        end = time.time()
    else:
        start = time.time()
        _ = model(dummy_data)
        end = time.time()
    forward_time = end - start
    model.to(device)
    return forward_time


def get_model_summary(model, input_shape, net_class='cnn'):
    """
    Get model summary using torchsummary library.

    Args:
        model
        input_shape
        net_class

    Returns:
        A dict containing following information:
            n_params: number of total parameters
            ops: number of operations
            size: model size
    """
    # Model eval
    model.eval()

    # Generate random input
    input_shape = (1, ) + input_shape
    model_input = torch.randn(input_shape)

    # Model forward
    if (net_class == 'rnn'):
        hidden_size =  model.init_hidden(1, torch.device('cpu')).data.numpy().shape
        h = torch.randn(hidden_size)
        info = summary(model, input_data=[model_input, h], verbose=0)
    else:
        info = summary(model, input_data=[model_input], verbose=0)
    
    # Store metrics
    n_params = info.total_params
    ops = info.total_mult_adds
    size = round(info.to_megabytes(info.total_input) 
                 + (info.total_output_bytes + info.total_param_bytes)/1e6, 2)

    model_summary = {
        'number_of_parameters': n_params,
        'number_of_operations': ops,
        'model_size_MB' : size
    }

    return model_summary


def save_train_metrics_plot(metrics, figure_path):
    """
    Plot loss and accuracy metrics generated 
    on a training
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.plot(metrics['train_loss'], label='training loss')
    plt.plot(metrics['validation_loss'], label='validation loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.subplot(122)
    plt.plot(metrics['train_accuracy'], label='training accuracy')
    plt.plot(metrics['validation_accuracy'], label='validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(figure_path)