import torch


def test_crop_pad_and_stretch_bring_a_picture_to_the_size():
    from modules.media.load.nodes import FunUIFitPicture
    wide = torch.ones(1, 40, 80, 3)
    for fit in ("crop", "pad", "stretch"):
        out = FunUIFitPicture.execute(wide, 64, 64, fit).result[0]
        assert out.shape == (1, 64, 64, 3), fit
    padded = FunUIFitPicture.execute(wide, 64, 64, "pad").result[0]
    assert float(padded[0, 0, 0, 0]) == 0.0 and float(padded[0, 32, 32, 0]) == 1.0          # black bars above and below, the picture between
    cropped = FunUIFitPicture.execute(wide, 64, 64, "crop").result[0]
    assert float(cropped.min()) == 1.0                                                      # nothing padded
