# Torch Delaunay - The Delaunay triangulation library for PyTorch

This is a fast library for computing [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation)
of 2-dimensional points.

The implementation is based on a sweep-algorithm, introduced by David Sinclair[^1] and later
improved by Volodymyr Agafonkin[^2].

## Installation

The library is distributed as PyPI package, to install the package, execute the following
command:
```sh
pip install torch_delaunay
```

You can use the `torch_delaunay` library for a fast computation of Delaunay triangulation for
points defined as [PyTorch](https://pytorch.org) tensors.

## Usage

```py
import torch
from torch_delaunay.functional import shull2d

# Compute Delaunay triangulation for randomly-generated 2-dimensional points.
points = torch.rand((100, 2))
simplices = shull2d(simplices)
```

## License

The Torch Delaunay is distributed under GPLv3 license. See the [LICENSE](LICENSE) file for full
license text.


[1]: David Sinclair - [S-hull: a fast radial sweep-hull routine for Delaunay triangulation](https://arxiv.org/abs/1604.01428).
[2]: Volodymyr Agafonkin - [Delaunator: An incredibly fast and robust JavaScript library for Delaunay triangulation of 2D points](https://github.com/mapbox/delaunator).
