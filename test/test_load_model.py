import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_models import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
seq_len = 4000
inp = torch.randn(batch_size, 1, seq_len).to(device)


@torch.no_grad()
def test_load_model():
    crnn_model, seq_lab_model = load_model(which="both")
    assert isinstance(crnn_model, torch.nn.Module)
    assert isinstance(seq_lab_model, torch.nn.Module)
    crnn_model = load_model(which="crnn")
    assert isinstance(crnn_model, torch.nn.Module)
    seq_lab_model = load_model(which="seq_lab")
    assert isinstance(seq_lab_model, torch.nn.Module)

    crnn_model.to(device)
    crnn_model.eval()
    seq_lab_model.to(device)
    seq_lab_model.eval()

    crnn_out = crnn_model(inp)
    seq_lab_out = seq_lab_model(inp)
    assert crnn_out.shape == (batch_size, 3)  # 3 classes
    assert seq_lab_out.shape == (batch_size, seq_len // 8, 2)
