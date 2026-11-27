"""本轮唯一B工厂的严格恢复与训练参数范围。"""
from pathlib import Path
import torch
from research.mentor_human_model.model.network import MentorHumanModel

ROOT = Path(__file__).resolve().parents[2]


def load_b_model(checkpoint, device='cpu'):
    """补入有明确源权重的冻结骨干后严格加载全部键，不静默丢参。"""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = {k: v for k, v in saved['config'].items() if k != 'prior_backend'}
    for key in ('base_weights', 'ft_checkpoint'):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    model = MentorHumanModel(**config)
    expected = model.state_dict()
    trained = saved['model']
    expected_trained = {k for k in expected if not k.startswith('vggt.aggregator.')}
    if set(trained) != expected_trained:
        raise ValueError({'missing': sorted(expected_trained-set(trained)),
                          'unexpected': sorted(set(trained)-expected_trained)})
    full = {k: v for k, v in expected.items() if k.startswith('vggt.aggregator.')}
    full.update(trained)
    model.load_state_dict(full, strict=True)
    model.to(device)
    return model, saved


def geometry_optimizer(model, options):
    """保持原几何模块可训，冻结骨干及外观；本B没有camera头。"""
    model.requires_grad_(True)
    model.vggt.aggregator.requires_grad_(False).eval()
    model.appearance.requires_grad_(False)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **options)
