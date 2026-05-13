import torch
from src.model import TwoHeadModel

def test_model_output_shape():
    model = TwoHeadModel()
    x = torch.ones((1, 3, 112, 112))
    y1, y2 = model(x)
    assert y1.shape == (1, 1)
    assert y2.shape == (1, 1)

