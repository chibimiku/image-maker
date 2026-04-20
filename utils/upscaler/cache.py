from collections import OrderedDict

from PIL import Image


class ImageCache:
    def __init__(self):
        self._data: OrderedDict[tuple, Image.Image] = OrderedDict()

    def get(self, key: tuple) -> Image.Image | None:
        value = self._data.pop(key, None)
        if value is not None:
            self._data[key] = value
        return value

    def put(self, key: tuple, image: Image.Image, max_items: int):
        if key in self._data:
            self._data.pop(key, None)
        self._data[key] = image

        while len(self._data) > max_items:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()
