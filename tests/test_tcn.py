import torch
from src.models.tcn import TCN


def test_tcn_forward_pass():
    """
    Test that the TCN accepts (batch, 48, 256) input and produces
    the correct output shapes for detection and classification.
    """

    model = TCN(
        input_channels=48,
        num_classes=4,
        tcn_channels=[64, 64, 128, 128],
        kernel_size=7,
        dropout=0.2
    )

    x = torch.randn(2, 48, 256)
    det, cls = model(x)

    assert det.shape == (2, 2), f"Expected (2, 2), got {det.shape}"
    assert cls.shape == (2, 4), f"Expected (2, 4), got {cls.shape}"


def test_tcn_single_sample():
    """Test with a single sample input."""

    model = TCN(
        input_channels=48,
        num_classes=4,
        tcn_channels=[64, 64, 128, 128],
        kernel_size=7
    )

    x = torch.randn(1, 48, 256)
    det, cls = model(x)

    assert det.shape == (1, 2)
    assert cls.shape == (1, 4)


def test_tcn_receptive_field():
    """Test that receptive field calculation works."""

    model = TCN(
        input_channels=48,
        num_classes=4,
        tcn_channels=[64, 64, 128, 128],
        kernel_size=7
    )

    rf = model.get_receptive_field()
    assert rf > 0, "Receptive field should be positive"
    # For [64,64,128,128] with kernel_size=7:
    # RF = 1 + 2*(7-1)*(1+2+4+8) = 1 + 12*15 = 181
    assert rf == 181, f"Expected receptive field of 181, got {rf}"


def test_cnn_forward_pass():
    """Test CNN baseline forward pass shapes."""

    from src.models.cnn_baseline import CNN1D

    model = CNN1D(
        input_channels=48,
        num_classes=4,
        dropout=0.2
    )

    x = torch.randn(2, 48, 256)
    det, cls = model(x)

    assert det.shape == (2, 2), f"Expected (2, 2), got {det.shape}"
    assert cls.shape == (2, 4), f"Expected (2, 4), got {cls.shape}"


def test_lstm_forward_pass():
    """Test LSTM baseline forward pass shapes."""

    from src.models.lstm_baseline import LSTMClassifier

    model = LSTMClassifier(
        input_features=48,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        dropout=0.2
    )

    x = torch.randn(2, 48, 256)
    det, cls = model(x)

    assert det.shape == (2, 2), f"Expected (2, 2), got {det.shape}"
    assert cls.shape == (2, 4), f"Expected (2, 4), got {cls.shape}"