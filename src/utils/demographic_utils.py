# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of EdgeWuW.
#
# EdgeWuW is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# EdgeWuW is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with EdgeWuW. If not, see https://www.gnu.org/licenses/

def filter_by_class(segmented_ids, results_df):
    filtered_dict = {}
    for key in segmented_ids:
        filtered_dict[key] = results_df[results_df.index.isin(segmented_ids[key])]
    return filtered_dict


def performance_gap(metric_1,metric_2):
    performance_gap = abs(metric_1-metric_2) * 100
    return performance_gap


#From -1 to 1, with 0 being ideal
def class_imbalance(advantaged_class_count,other_class_count):
    class_imbalance = (advantaged_class_count-other_class_count)/(advantaged_class_count+other_class_count)
    return class_imbalance


def get_ids_by_class(dataframe, group):
    # Get ids by a specific group
    ids = {}

    # Get unique values
    unique_values = dataframe[group].unique()

    # Generate IDs for each category
    for value in unique_values:
        ids[value] = dataframe[dataframe[group] == value].index.tolist()

    return ids
