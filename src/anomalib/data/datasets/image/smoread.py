from collections.abc import Sequence
from pathlib import Path
import os

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.utils import deprecate

class SmoreADDataset(AnomalibDataset):
    """Smore AD dataset class.

    Dataset class for loading and processing SMore AD dataset images from text files. Supports
    both classification and segmentation tasks.

    Args:
        txt_path (Path | str): Path to text file containing the dataset paths and labels.
            Defaults to ``"./datasets/MVTecAD"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import SmoreADDataset
        >>> dataset = SmoreADDataset(
        ...     txt_path=Path("./datasets/train.txt"),
        ...     split="train"
        ... )

        For classification tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask paths and masks:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        txt_path: Path | str = "./datasets/MVTecAD",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.txt_path = Path(txt_path)
        self.split = split
        self.delimeter = ","
        self.samples = make_ad_dataset(
            self.txt_path,
            split=self.split,
            delimeter=self.delimeter
        )

def make_ad_dataset(
    txt_path: str | Path,
    split: str | Split | None = None,
    delimeter: str | None = None
) -> DataFrame:
    """Create SMore AD samples by parsing the text file structure.

    The text file is expected to follow the format:
        ``image_path,label``
        ``/path/to/image1.jpg,good``
        ``/path/to/image2.jpg,defect``

    Args:
        txt_path (str | Path): Path to text file containing dataset paths and labels
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        delimeter (str | None, optional): Delimiter used in the text file
            Defaults to ``","``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> txt_path = Path("./datasets/train.txt")
        >>> samples = make_ad_dataset(txt_path, split="train")
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  /path/to/dataset train good  [...]/image1.jpg           0
        1  /path/to/dataset train good  [...]/image2.jpg           0

    Raises:
        RuntimeError: If no valid images are found in the text file
    """

    txt_path = validate_path(txt_path)
    txt_lines = open(txt_path, "r").readlines()
    samples_list = []
    for line in txt_lines:
        line = line.strip()
        img_path = line.split(delimeter)[0]
        label = line.split(delimeter)[1]
        samples_list.append({"path": os.path.dirname(img_path), "split": split, "label": label, "image_path": img_path, "mask_path": ""})

    if not samples_list:
        msg = f"Found 0 images in {txt_path}"
        raise RuntimeError(msg)

    samples = DataFrame(samples_list, columns=["path", "split", "label", "image_path", "mask_path"])
    # Create label index for normal (0) and anomalous (1) images.
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples