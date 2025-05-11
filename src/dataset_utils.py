import pathlib
from PIL import Image
import skimage
import urllib.request
import shutil

class DatasetManager:
    def __init__(self, directory, **kwargs):
        self.image_suffix = kwargs.pop('image_suffix', '.jpg')
        self.gold_suffix = kwargs.pop('gold_suffix', '.tif')
        self.mask_suffix = kwargs.pop('mask_suffix', '.tif')

        self.dirs = {'main': pathlib.Path(directory)}
        for subdir in ['image', 'gold', 'mask']:
            self.dirs[subdir] = self.dirs['main'] / subdir

        for dir in self.dirs.values():
            assert dir.exists() and dir.is_dir()

        self.stems = sorted(f.stem for f in self.dirs['image'].glob(f'*{self.image_suffix}'))
        gold_stems = sorted(f.stem for f in self.dirs['gold'].glob(f'*{self.gold_suffix}'))
        mask_stems = sorted(f.stem for f in self.dirs['mask'].glob(f'*{self.mask_suffix}'))

        assert self.stems == gold_stems == mask_stems

        sample_image = Image.open(self.dirs['image'] / f'{self.stems[0]}{self.image_suffix}')
        sample_gold = Image.open(self.dirs['gold'] / f'{self.stems[0]}{self.gold_suffix}')
        sample_mask = Image.open(self.dirs['mask'] / f'{self.stems[0]}{self.mask_suffix}')

        self.width, self.height = sample_image.size
        assert sample_gold.size == (self.width, self.height)
        assert sample_mask.size == (self.width, self.height)
        self.image_channels = len(sample_image.getbands())

        for i in range(1, len(self.stems)):
            image = Image.open(self.dirs['image'] / f'{self.stems[i]}{self.image_suffix}')
            gold = Image.open(self.dirs['gold'] / f'{self.stems[i]}{self.gold_suffix}')
            mask = Image.open(self.dirs['mask'] / f'{self.stems[i]}{self.mask_suffix}')

            assert image.size == (self.width, self.height)
            assert gold.size == (self.width, self.height)
            assert mask.size == (self.width, self.height)
            assert len(image.getbands()) == self.image_channels

        self.dataset_size = len(self.stems)

    def load(self, subdir, entry_id):
        if subdir == 'image':
            path = self.dirs[subdir] / f'{self.stems[entry_id]}{self.image_suffix}'
            return skimage.util.img_as_float32(skimage.io.imread(path))
        elif subdir == 'gold':
            path = self.dirs[subdir] / f'{self.stems[entry_id]}{self.gold_suffix}'
            return skimage.util.img_as_bool(skimage.io.imread(path, as_gray=True))
        elif subdir == 'mask':
            path = self.dirs[subdir] / f'{self.stems[entry_id]}{self.mask_suffix}'
            return skimage.util.img_as_bool(skimage.io.imread(path, as_gray=True))
        return None

def get_resized_image(image, factor):
    assert factor > 0.0
    new_size = (int(image.width * factor), int(image.height * factor))
    return image.resize(new_size, resample=Image.LANCZOS)

def prepare_dataset(directory, **kwargs):
    resize_factor = kwargs.pop('resize_factor', None)

    main_dir = pathlib.Path(directory)
    main_dir.mkdir(exist_ok=True)
    dataset_zip = main_dir / 'dataset.zip'

    for item in main_dir.iterdir():
        if item == dataset_zip:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    if not dataset_zip.exists():
        url = 'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip'
        urllib.request.urlretrieve(url, dataset_zip)

    shutil.unpack_archive(dataset_zip, main_dir)

    pathlib.Path(main_dir / 'images').rename(main_dir / 'image')
    pathlib.Path(main_dir / 'manual1').rename(main_dir / 'gold')

    for item in pathlib.Path(main_dir / 'mask').iterdir():
        new_name = item.stem[:-5] + item.suffix
        item.rename(item.with_name(new_name))

    if resize_factor is not None:
        for subdir in main_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.iterdir():
                    resized = get_resized_image(Image.open(file), resize_factor)
                    resized.save(file)
