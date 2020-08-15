# -*- coding: utf-8 -*-
import torch
from evalib.utils import get_device

__author__ = "Ramjee Ganti"
__copyright__ = "Ramjee Ganti"
__license__ = "new-bsd"


def test_get_device():
    assert get_device() == torch.device("cpu")
