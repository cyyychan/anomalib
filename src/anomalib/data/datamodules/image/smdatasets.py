# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SMore AD Data Module.

This module provides a PyTorch Lightning DataModule for the SMore AD dataset.
The dataset is loaded from text files containing image paths and labels.

Example:
    Create a SMore AD datamodule::

        >>> from anomalib.data import SmoreAD
        >>> datamodule = SmoreAD(
        ...     train_txt="./datasets/train.txt",
        ...     test_txt="./datasets/test.txt"
        ... )

Notes:
    The dataset expects text files with the following format:
        image_path,label
        /path/to/image1.jpg,good
        /path/to/image2.jpg,defect

    The directory structure should be::

        datasets/
        ├── train.txt
        └── test.txt
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image import SmoreADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.utils import deprecate

logger = logging.getLogger(__name__)

class SmoreAD(AnomalibDataModule):
    """SMore AD Datamodule.

    Args:
        train_txt (Path | str): Path to the training text file containing image paths and labels.
            Defaults to ``"./datasets/MVTecAD/train.txt"``.
        test_txt (Path | str): Path to the test text file containing image paths and labels.
            Defaults to ``"./datasets/MVTecAD/test.txt"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create SMore AD datamodule with default settings::

            >>> datamodule = SmoreAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Use custom text files::

            >>> datamodule = SmoreAD(
            ...     train_txt="./datasets/custom_train.txt",
            ...     test_txt="./datasets/custom_test.txt"
            ... )

        Create validation set from test data::

            >>> datamodule = SmoreAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set::

            >>> datamodule = SmoreAD(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    def __init__(
        self,
        train_txt: Path | str = "./datasets/MVTecAD/train.txt",
        test_txt: Path | str = "./datasets/MVTecAD/test.txt",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.train_txt = Path(train_txt)
        self.test_txt = Path(test_txt)

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = SmoreADDataset(
            split=Split.TRAIN,
            txt_path=self.train_txt
        )
        self.test_data = SmoreADDataset(
            split=Split.TEST,
            txt_path=self.test_txt
        )
