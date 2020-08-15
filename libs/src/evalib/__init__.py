# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version

    from . import data  # noqa
    from . import datasets  # noqa
    from . import gradcam  # noqa
    from . import lr  # noqa
    from . import models  # noqa
    from . import plot  # noqa
    from . import tnt  # noqa
    from . import utils  # noqa

    __all__ = ["utils", "plot", "data", "models", "gradcam", "tnt"]
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
