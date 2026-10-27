# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import os
import json
import pandas as pd
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
#from torch.utils.tensorboard.summary import hparams

import keyword_spotting.loader as loader
import keyword_spotting.trainer as trainer
from keyword_spotting.feature_extractor import FeatureExtractor
from keyword_spotting.arguments import parse_arguments
from keyword_spotting.reporter import Reporter
from utils.metrics import get_metrics, plot_confusion_matrix
from utils.demographic_utils import get_ids_by_class, filter_by_class
from utils.experiment_utils import *


def main():
    #########################
    #   Setup experiment    #
    #########################
    
    args = parse_arguments()

    # Model and experiment identification
    model_id = args.arc
    architecture_path = os.path.join(args.checkpoint, 'kws_' + args.arc)
    version = args.version if args.version else check_version(architecture_path)
    experiment_path = os.path.join(architecture_path, 'version_' + str(version))
    model_path = os.path.join(experiment_path, model_id + '_entire.pt')

    print("Testing model: " + model_id + "\n")
    csv_path = os.path.join(experiment_path, 'csv')
    images_path = os.path.join(experiment_path, 'images')
    check_path(csv_path)
    check_path(images_path)

    # Reporter
    reporter = Reporter(experiment_path, model_id + '_report.json')
    reporter.load(os.path.join(experiment_path, model_id + '_report.json'))

    train_args = reporter.data['arguments']

    # Set seed
    trainer.set_seed(args.seed)

    # Select device
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device("cuda")
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Summary 
    """
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
    """

    warnings.filterwarnings("ignore")

    #########################
    #      Evaluation       #
    #########################

    # Build WuW model
    kws_model = torch.load(model_path)
    kws_model.to(device)

    # Net type
    if("sgru" in args.arc):
        net_class = 'rnn'
    else:
        net_class = 'cnn'

    # Classification Loss function
    if(args.one_hot):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    # Set feature extractor
    fe = FeatureExtractor(
        device, 
        train_args['time_window'],  
        train_args['sampling_rate'], 
        train_args['feature_extractor'], 
        train_args['transpose']
        )

    # FIXME: this was done in the past to test Speech Enhancement models
    # We may want to separte this behaviour in a separate script
    se_models = [None] # [WuW, SE+WuW]

    # Test WuW and SE+WuW
    for se_model in se_models:

        print('\n\nSNR range: [' + str(args.min_snr) + ', ' + str(args.max_snr)+'] using SE: ' + str(se_model))

        #########################
        #     Data Loaders      #
        #########################

        # NOTE: the test must be done also with the dev set
        key = 'test'
        plot_graphs = True

        if args.specific_group:
            dataset_name = args.test_filename
            test_name = dataset_name.rsplit(".", 1)[0]
        else:
            dataset_name = 'test.tsv'
            test_name = 'test_metrics_' + str(args.min_snr) + '_' + str(args.max_snr)

        if(se_model != None):
            test_name = "se_" + test_name

        # Read test set
        test_dataset = loader.load_test_partition(
            args.partition_path,
            window_size=int(args.time_window*args.sampling_rate),
            fs=args.sampling_rate,
            snr_range=[args.min_snr, args.max_snr],
            bn_prob=1.0,
            non_speech_prob=0.0,
            rir_prob=0.0,
            time_stretch=False,
            dataset_name=dataset_name
        )

        # Generate test dataframe
        test_dataframe = test_dataset.dataframe
        test_dataframe.set_index('Sample_ID', inplace=True, drop=False)

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

        # Get classes index
        target_names = []
        dict_path = os.path.join(args.partition_path, 'classes_index.json')
        classes_index = json.load(open(dict_path, 'r'))
        n_clases = len(classes_index)
        for index in range(n_clases):
            target_names.append(classes_index[str(index)])

        # Save results dataframe with ids
        results_tsv_path = os.path.join(csv_path, test_name + '_results.tsv')
        results_df = save_scores_and_labels(results_tsv_path, labels, predicted_scores, ids, target_names)

        
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
        
        reporter.report("results"+"_"+dataset_name.rsplit(".", 1)[0], result[1])
        
        # Tensorboard test results
        """
        tensorboard_writer.add_scalar("test_loss", metrics['test_loss'])
        tensorboard_writer.add_scalar("test_accuracy", metrics['test_accuracy'])
        tensorboard_writer.add_scalar("macro-avg-f1-score", result[1]['macro avg']['f1-score'])
        tensorboard_writer.add_figure("confussion_matrix", cm_fig)
        tensorboard_writer.close()
        """

        if args.specific_group:
            # Get segmented indices
            segmented_ids = get_ids_by_class(test_dataframe, args.specific_group)

            # Filter by class
            filtered_dict = filter_by_class(segmented_ids, results_df)

            specific_group_report = {}

            for class_key, value in filtered_dict.items():
                labels = value.loc[:,"True_Label"].tolist()
                predicted_scores = [value.loc[:,target_names[0]].tolist(),value.loc[:,target_names[1]].tolist()]
                predicted_scores = list(map(list, zip(*predicted_scores)))

                if len(predicted_scores):
                    result_by_class, _ = get_metrics(
                        labels,
                        predicted_scores,
                        images_path,
                        target_names=target_names,
                        optimal_thresholds=None,
                        plot_graphs=plot_graphs
                        )
                    """
                    cm_fig = plot_confusion_matrix(
                        result_by_class[0],
                        target_names,
                        os.path.join(images_path, test_name + '_confusion_' + str(class_key) +'.png'),
                        cmap='BuPu',
                        normalize=True
                        )
                    """
                else:
                    result_by_class = [0,"Not enough samples"]

                specific_group_report[str(class_key)] = result_by_class[1]

            reporter.report(test_name + "_" + args.specific_group, specific_group_report)

            # Append Specific group column to results dataframe
            test_dataframe.set_index('Sample_ID', inplace=True, drop=False)
            results_df = pd.concat([results_df, test_dataframe[args.specific_group]], axis=1)

            # Save results dataframe with ids
            results_df.to_csv(results_tsv_path, index=False, sep='\t')


    print("Finished testing, results saved in {0}".format(os.path.join(experiment_path, model_id + '_report.json')))

if __name__ == "__main__":
    main()
