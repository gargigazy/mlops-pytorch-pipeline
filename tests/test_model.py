# basic tests for the model
# run with: pytest tests/

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from model import get_model, SimpleCNN


def test_resnet18_output_shape():
    # model should output 10 classes for cifar10
    model = get_model("resnet18", num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)  # batch of 2 fake images
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_unknown_architecture_raises_error():
    # should throw error for a model name that doesnt exist
    with pytest.raises(ValueError):
        get_model("some_fake_model", num_classes=10)


def test_model_trains_one_step():
    # check that loss goes down er atleast that backprop works without crashing
    model = get_model("simple_cnn", num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0  # loss should be a positive number
