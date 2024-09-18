import torch
from torchmetrics import KLDivergence, MetricCollection
from torchmetrics.regression import ExplainedVariance, MeanSquaredError, R2Score


def initialize_metrics(args, prefix="train_"):
    """Get the metrics to be used in the training loop."""
    # Define metrics and move to GPU if available
    metrics = {
        "MSE": MeanSquaredError(),
        "R2": R2Score(),
        "RMSE (℃)": MeanSquaredError(squared=False),
        # "ExplainedVariance": ExplainedVariance(),
        # "KLDivergence": KLDivergence(log_prob=True),
    }
    [metric.to(args.device) for name, metric in metrics.items()]
    mc = MetricCollection(metrics, prefix=prefix)
    return mc.to(args.device)


def convert_metrics_to_scalar(m, train_or_val):
    return {
        f"{train_or_val}/{k}": v.cpu().detach().numpy()
        for k, v in m.items()
        if isinstance(v, torch.Tensor) and v.ndim <= 1 and v.numel() == 1
    }
