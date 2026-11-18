# Applied-Lab Dataset Provenance

Datasets are external research artifacts. The repository does not redistribute them.

The word **public** here means publicly downloadable for research/education. Dataset terms are separate from this repository's Apache-2.0 software license. Learners are responsible for following the dataset's own terms.

## CIFAR-10

Used by Applied Lab 01 and potentially later representation-learning labs.

- Canonical source: https://cave.cs.toronto.edu/kriz/cifar.html
- Source: University of Toronto CIFAR dataset
- Loader: `torchvision.datasets.CIFAR10`
- Torchvision API: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- Contents: 60,000 32×32 color images across 10 classes
- Standard split: 50,000 train / 10,000 test
- Download behavior: the lab uses torchvision's `download=True` path when requested
- Repository policy: data remains under the user-selected local dataset directory and is never committed

Recommended local path:

```text
data/cifar10/
```

The applied lab creates its own deterministic validation split from the official training split.

## Future labs

Before a new dataset is added, record:

1. canonical source
2. task
3. expected size
4. train/validation/test policy
5. download mechanism
6. dataset terms/license URL when available
7. whether redistribution is permitted
8. checksum/version if the upstream project exposes one

Possible future datasets include Penn-Fudan Pedestrian, Oxford-IIIT Pet, and other small public datasets exposed through torchvision or canonical project pages.
