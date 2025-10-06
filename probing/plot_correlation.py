import torch
import matplotlib.pyplot as plt


def plot_feature_correlation(X: torch.Tensor):
    corr = torch.corrcoef(X.T)
    plt.imshow(corr.abs() ** 0.5, cmap='viridis', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Correlation')
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig("plots/correlation", dpi=300)
    plt.close()