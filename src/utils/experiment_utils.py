# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

import os
import numpy as np
import pandas as pd


def check_version(architecture_path):
    version = 0
    if os.path.isdir(architecture_path):
        subfolders = next(os.walk(architecture_path))[1]
        versions = []

        for x in subfolders:
            if 'version' in x:
                versions.append(int(float(x.replace('version_', ''))))

        version = max(versions)
    return version


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


def save_scores_and_labels(tsv_path, labels, scores, ids, target_names=None):
    # Merge all data in a single dataframe
    scores_df = pd.DataFrame(index=ids, data=np.array(scores))
    labels_df = pd.DataFrame(index=ids, data=np.array(labels))
    ids_df = pd.DataFrame(index=ids, data=np.array(ids))

    results_df = pd.concat([ids_df, labels_df, scores_df], axis=1)
    number_of_columns = len(results_df.columns)

    column_names = ['Sample_ID', 'True_Label'] + ['Score_' + str(i) for i in range(number_of_columns - 2)]

    if target_names is not None:
        column_names = ['Sample_ID', 'True_Label'] + target_names
    
    results_df.columns = column_names
    results_df.to_csv(tsv_path, index=False, sep='\t')
    return results_df