"""
Print the key metrics of multiple experiments to the standard output.

For more information, see: python main.py --help or have a look at the README.
"""

__author__ = "Kilian Batzner, Paul Bergmann, Michael Fauser, David Sattlegger"
__copyright__ = "2022, MVTec Software GmbH"

import argparse
import json
import os
from os.path import join

import numpy as np
from tabulate import tabulate

from evaluate_experiment import OBJECT_NAMES


def parse_user_arguments():
    """
    Parse user arguments.

    Returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument(
        '--metrics_folder',

        default=r"/data0/xmh/code/24code/EfficientAD/output/res_spro",
        # default="/data0/mhxu/3090_4icv/code/RD4AD-main/outputM",
        # default="/data0/mhxu/code/RD4AD-main/outputM",
        help='Path to the folder that contains the evaluation results.')

    parser.add_argument(
        '--metric_type',
        default='localization',
        choices=['localization', 'classification'],
        help='Print results for either anomaly localization or'
             ' classification.')

    parser.add_argument(
        '--integration_limit',
        default='0.05',
        help='For anomaly localization, report results at this integration'
             ' limit. Note that the value specified here also needs to be'
             ' present in the respective \'metrics.json\' file. Therefore,'
             ' it also needs to be specified as an integration limit in '
             ' \'evaluate_experiment.py\' during evaluation.')

    return parser.parse_args()


def extract_table_rows(metrics_folder, metric_type, integration_limit, mode):
    """
    Extract all rows to create a table that displays a given metric for each
    evaluated method.

    Args:
        metrics_folder:    Base folder that contains evaluation results.
        metric_type:       Type of the metric to be extracted. Choose between
                           'localization' and 'classification'.
        integration_limit: For anomaly localization, fetch results for a certain





















                           integration limit.
        mode:              Fetch metrics for logical anomalies, structural
                           anomalies, or the mean of both values.

    Returns:
        List of table rows. Each row contains the method name and the extracted
            metrics for each evaluated object as well as the mean performance.
    """
    assert metric_type in ['localization', 'classification']
    assert mode in ['logical_anomalies', 'structural_anomalies', 'mean']

    # Iterate each evaluated method.
    method_ids = os.listdir(metrics_folder)
    rows = []
    for method_id in method_ids:

        # Each row starts with the name of the method.
        row = [method_id]
        metric_values = []

        # Fetch metrics for each dataset object.
        skipped = False
        for obj in OBJECT_NAMES:

            # Open the metrics file.
            metrics_file = \
                join(metrics_folder, method_id, obj, 'metrics.json')

            # Skip this object if no numbers exist for it.
            if not os.path.exists(metrics_file):
                print(f"Warning: {metrics_file} does not exist.")
                row.append('-')
                skipped = True
                continue

            with open(metrics_file) as file:
                metrics = json.load(file)

            # Fetch the desired metric and store it in the table row.
            if metric_type == 'localization':
                metric = \
                    metrics[metric_type]['auc_spro'][mode][integration_limit]
            else:
                metric = metrics[metric_type]['auc_roc'][mode]
            metric_values.append(metric)

        # Compute mean performance if no object had to be skipped.
        row.extend(np.round(metric_values, decimals=3))

        if not skipped:
            mean_performance = np.mean(metric_values)
            row.append(np.round(mean_performance, decimals=3))
        else:
            row.append('-')

        rows.append(row)

    return rows


def main():
    """
    Print the key metrics of multiple experiments to the standard output.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    for mode in ['logical_anomalies', 'structural_anomalies', 'mean']:

        # Create the table rows. One row for each experiment.
        rows_pro_structural = \
            extract_table_rows(
                metrics_folder=args.metrics_folder,
                metric_type=args.metric_type,
                integration_limit=args.integration_limit,
                mode=mode)
        # Print localization result table.

        if args.metric_type == 'localization':
            info_string = f"\nAUC sPRO ({args.metric_type} -- {mode})"
            info_string += f" Integration limit: {args.integration_limit}"
        else:
            info_string = f"\nAUC ROC ({args.metric_type} -- {mode})"
        print(info_string)

        print(tabulate(rows_pro_structural,
                       headers=['Method'] + OBJECT_NAMES + ['Mean'],
                       tablefmt='fancy_grid'))


if __name__ == "__main__":
    main()
