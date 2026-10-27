# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import keyword_spotting.models as models
from keyword_spotting.arguments import parse_arguments
import keyword_spotting.loader as loader
from keyword_spotting.feature_extractor import FeatureExtractor
from keyword_spotting.reporter import Reporter
import keyword_spotting.trainer as trainer
import keyword_spotting.augments as augments
from utils.metrics import get_metrics, plot_confusion_matrix
from utils.experiment_utils import *

import datetime
import shutil
import json
import os


def main():

    #########################
    #   Setup experiment    #
    #########################

    # Parse arguments
    args = parse_arguments()

    # Model and experiment identification
    current_time = datetime.datetime.now()
    current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_id = args.arc
    architecture_path = os.path.join(args.checkpoint, 'kws_' + args.arc)
    version = check_version(architecture_path) + 1
    experiment_path = os.path.join(architecture_path, 'version_' + str(version))
    images_path = os.path.join(experiment_path, 'images')
    csv_path = os.path.join(experiment_path, 'csv')
    check_path(args.checkpoint) # Checkpoint path
    check_path(architecture_path) # Architecture path
    check_path(experiment_path) # Experiment path
    check_path(images_path) # Images path
    check_path(csv_path) # csv path

    # Print arguments and version
    print("\n\n***************************************")
    print("******* Experiment Information ********")
    print("***************************************")
    print("Model id: " + model_id)
    print("Version: " + str(version))
    print("Current time: " + current_time)
    print("Experiment path: " + experiment_path)
    print("Arguments:")
    print(vars(args))
    print("***************************************\n\n")

    # Set Report
    reporter = Reporter(experiment_path, model_id  + '_report.json')
    reporter.report('arguments', vars(args))
    reporter.report('experiment_date', current_time)

    # Set seed
    trainer.set_seed(args.seed)

    # Select device
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device("cuda:" + str(args.cuda_device))
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Summary 
    tensorboard_writer = SummaryWriter(experiment_path)
    tb_metrics_dict = {
        'best_val_loss':0,
        'best_epoch':0,
        'training_time':0,
        'macro-avg-f1-score':0,
        'test_loss':0,
        'test_accuracy':0,
    }
    exp, ssi, sei = hparams(vars(args), metric_dict=tb_metrics_dict)
    tensorboard_writer.file_writer.add_summary(exp)
    tensorboard_writer.file_writer.add_summary(ssi)                 
    tensorboard_writer.file_writer.add_summary(sei)

    #########################
    #     Data Loaders      #
    #########################

    # Read train and dev sets
    train_dataset, validation_dataset = loader.load_train_partitions(
        args.partition_path,
        window_size=int(args.time_window*args.sampling_rate),
        fs=args.sampling_rate,
        bn_prob=args.bn_prob,
        non_speech_prob=args.non_speech_prob,
        snr_range=[args.min_snr, args.max_snr],
        rir_prob=args.rir_prob,
        dir_prob=args.dir_prob,
        time_stretch=args.time_stretch
    )
    print("Train data information")
    print(train_dataset)

    # To deal with dataset unbalance: get underrepresented class
    # Read class index
    with open(os.path.join(args.partition_path, 'partition.json')) as f:
        partition_info = json.load(f)
    speech_samples = partition_info['Train']['Samples information']['speech']
    minimum_represented_class = min(speech_samples, key=speech_samples.get)
    n_minimum_samples = speech_samples[minimum_represented_class]

    # Generate DataLoaders for training
    sampler = None
    shuffle_train = True
    if(args.use_batch_sampler):
        sampler = torch.utils.data.WeightedRandomSampler(
            train_dataset.get_samples_weight(),
            len(speech_samples.keys())*n_minimum_samples, # ~ twice the size of wuw samples
            replacement=True # reapeats samples of underrepresented classes
            )
        shuffle_train = False

    # Set batch size
    batch_size = args.batch_size
    
    # Generate data loaders
    train_loader = DataLoader(
        train_dataset, 
        shuffle=shuffle_train, 
        batch_size=batch_size, 
        drop_last=False, 
        sampler=sampler,
        num_workers=args.num_workers
        )

    # NOTE: The data in the validation split is not balanced
    # Think about what happens when selecting the model with the best validation loss
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=args.num_workers
        )

    # Augments
    augments_list = []
    if (args.wn_prob > 0.0):
        augments_list.append(augments.WhiteNoise(
            min_snr_db=args.wn_min_snr,
            max_snr_db=args.wn_max_snr,
            p=args.wn_prob
            ))
    if (args.freqmask_prob > 0.0):
        augments_list.append(augments.FrequencyMasking(
            n_fft=args.augment_nfft,
            hop_length=args.augment_hop_len,
            freq_mask_param=args.freqmask_max_len,
            n_freq_masks=args.freqmask_n_masks,
            p=args.freqmask_prob
            ))
    if (args.timemask_prob > 0.0):
        augments_list.append(augments.TimeMasking(
            n_fft=args.augment_nfft,
            hop_length=args.augment_hop_len,
            time_mask_param=args.timemask_max_len,
            n_time_masks=args.timemask_n_masks,
            p=args.timemask_prob
            ))
    if (args.timestretch_prob > 0.0):
        augments_list.append(augments.TimeStretching(
            n_fft=args.augment_nfft,
            hop_length=args.augment_hop_len,
            target_size=int(args.time_window*args.sampling_rate),
            eps=args.timestretch_eps,
            p=args.timestretch_prob
            ))
    if (args.randaugment_prob > 0.0):
        augments_list.append(augments.AudioRandAugment(
            n_fft=args.augment_nfft,
            hop_length=args.augment_hop_len,
            num_ops=args.randaugment_n_ops,
            magnitude=args.randaugment_magnitude,
            p=args.randaugment_prob,
            target_size=int(args.time_window*args.sampling_rate),
            num_magnitude_bins=args.randaugment_n_mag_bins
            ))
    if (args.fade_prob > 0.0):
        augments_list.append(augments.Fade(
            p=args.fade_prob,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
            fade_shape=args.fade_shape
        ))
    if (args.filteraugment_prob > 0.0):
        augments_list.append(augments.FilterAugment(
            p=args.filteraugment_prob,
            db_range=(args.filteraugment_min_db, args.filteraugment_max_db),
            n_band=(args.filteraugment_min_bands, args.filteraugment_max_bands),
            min_bw=args.filteraugment_min_bw,
            filter_type=args.filteraugment_filter_type,
            n_fft=args.augment_nfft,
            hop_length=args.augment_hop_len
        ))
    data_augment = torch.nn.Sequential(*augments_list)
    data_augment.to(device)

    # Label-based augments: none in this release (identity)
    label_based_augment = models.TwoInputSequential()
    label_based_augment.to(device)

    # Set feature extractor
    fe = FeatureExtractor(device, args.time_window, args.sampling_rate, args.feature_extractor, args.transpose)
    fe_parameters = dict((key, str(value)) for key, value in fe.__dict__.items() if not callable(value) and not key.startswith('_'))
    reporter.report('feature_extractor', fe_parameters)

    #FIXME: Temporary fix: FreqMixStyle is applied in the frequency dimension directly
    if (args.freqmixstyle_prob > 0.0):
        freq_mix_style = augments.FreqMixStyle(
            p=args.freqmixstyle_prob,
            alpha=args.freqmixstyle_alpha,
        )
        freq_mix_style.to(device)
    else:
        freq_mix_style = None

    #########################
    #       Training        #
    #########################
    
    # Build model
    model = models.SerializableModule().create(args.arc, fe.output_shape, num_classes=train_dataset.n_classes)
    model = model.to(device)

    # Check if start from checkpoint
    if(args.from_checkpoint != ""):
        model.load(os.path.join(
            args.from_checkpoint,
            'kws_' + args.arc,
            model_id + '.pt'
            ))
    
    # Net class: rnn or cnn
    net_class = 'rnn' if("sgru" in args.arc) else 'cnn'

    # Get model summary
    model_summary = trainer.get_model_summary(model, fe.output_shape, net_class)
    reporter.report('model_summary', model_summary)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif(args.optimizer.lower() == 'rmsprop'):
        optimizer = torch.optim.RMSprop(
            model_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
            )
    elif('sgd' in args.optimizer.lower()):
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
            )

    # Model path
    model_path = os.path.join(experiment_path, model_id)
    
    # Train model
    model, metrics = trainer.train_kws(
        model,
        fe,
        data_augment,
        label_based_augment,
        train_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        tensorboard_writer,
        args.epochs,
        batch_size,
        model_path,
        lr_patience=args.lr_patience,
        patience=args.patience,
        net_class=net_class,
        one_hot=args.one_hot,
        freq_mix_style=freq_mix_style
        )

    # Save plot and metrics
    reporter.report('train_metrics', metrics)
    figure_path = os.path.join(images_path,  'train_metrics.png')
    trainer.save_train_metrics_plot(metrics, figure_path)

    # Store class index in experiment path
    shutil.copyfile(os.path.join(args.partition_path, 'classes_index.json') , os.path.join(experiment_path, 'classes_index.json'))

    #########################
    #   Evaluation stage    #
    #########################

    # Set test seed
    trainer.set_seed(args.test_seed)

    se_model = None
    kws_model = torch.load(model_path + '_entire.pt')
    kws_model.to(device)

    print("\n\n\nTesting model: " + model_id + "\n")
    test_name = 'test_metrics_' + str(args.min_snr) + '_' + str(args.max_snr)

    # NOTE: the test must be don also with the dev set
    key = 'test'
    plot_graphs = True

    # Read test set
    test_dataset = loader.load_test_partition(
        args.partition_path,
        window_size=int(args.time_window*args.sampling_rate),
        fs=args.sampling_rate,
        snr_range=[args.min_snr, args.max_snr],
        bn_prob=1.0,
        non_speech_prob=0.0,
        rir_prob=0.0,
        dir_prob=0.0,
        time_stretch=False
    )

    print("Test data information")
    print(test_dataset)

    # Generate data loaders
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        drop_last=False
    )

    # Test model: adapt to return multiclass vector
    ids, labels, predicted_scores, predicted_labels, metrics = trainer.test_kws(
        kws_model,
        fe,
        test_loader,
        criterion,
        device,
        args.batch_size,
        net_class=net_class,
        one_hot=args.one_hot,
        se_model=se_model
    )

    # Save results dataframe
    save_scores_and_labels(os.path.join(csv_path, test_name + '.tsv'), labels, predicted_scores, ids)

    # Get classes index
    target_names = []
    dict_path = os.path.join(args.partition_path, 'classes_index.json')
    classes_index = json.load(open(dict_path, 'r'))
    for index in range(len(predicted_scores[0])):
        target_names.append(classes_index[str(index)])

    # Analyze results
    result, _ = get_metrics(
        labels,
        predicted_scores,
        images_path,
        target_names=target_names,
        optimal_thresholds=None,
        plot_graphs=plot_graphs
        )
    cm_fig = plot_confusion_matrix(
        result[0],
        target_names,
        os.path.join(images_path, test_name + '_confusion_' + key +'.png'),
        cmap='BuPu',
        normalize=True
        )

    # Store metrics
    test_metrics = {
        'loss':metrics['test_loss'],
        'accuracy':metrics['test_accuracy'],
        'report': result[1]
    }
    reporter.report(test_name, test_metrics)

    # Tensorboard test results
    tensorboard_writer.add_scalar("test_loss", metrics['test_loss'])
    tensorboard_writer.add_scalar("test_accuracy", metrics['test_accuracy'])
    tensorboard_writer.add_scalar("macro-avg-f1-score", result[1]['macro avg']['f1-score'])
    tensorboard_writer.add_figure("confussion_matrix", cm_fig)
    tensorboard_writer.close()
    
if __name__ == "__main__":
    main()
