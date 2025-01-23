import os
import numpy as np

def hp_range():
    hp_dir = './results/hp'
    if not os.path.exists(hp_dir):
        os.makedirs(hp_dir)

    hidden_dim1_range = [150]
    hidden_dim2_range = [75]
    lr_range = [1e-4, 1e-3]
    datasets = ['MSLR30K', 'Yahoo', 'ISTELLA']
    points = [100, 1000, 10000, 100000]

    for dataset in datasets:
        for point in points:
            for hidden_dim1 in hidden_dim1_range:
                for hidden_dim2 in hidden_dim2_range:
                    for lr in lr_range:
                        # Generate filenames for both ips and risk methods
                        filename_ips = f"{dataset}_{point}_{lr}_ips"
                        filepath_ips = os.path.join(hp_dir, filename_ips)
                        with open(filepath_ips, 'w') as f:
                            f.write(str(np.random.rand()))  # Example validation DCG value

                        filename_risk = f"{dataset}_{point}_{lr}_risk"
                        filepath_risk = os.path.join(hp_dir, filename_risk)
                        with open(filepath_risk, 'w') as f:
                            f.write(str(np.random.rand()))  # Example validation DCG value

if __name__ == '__main__':
    hp_range()