
import os
import uuid

from decimal import Decimal

from django.db import models


class Prediction(models.Model):
    def image_path(self, filename: str) -> str:
        """
        Generates the file path to store the image
        :param filename: name of the image file
        :return: path where the image will be stored
        """
        ext = filename.split('.')[-1]
        file = 'header_image.' + ext
        return os.path.join('images_files', self.prediction, file)

    image = models.ImageField(null=False, blank=False, upload_to=image_path, default='images/images.png')
    prediction = models.CharField(max_length=50, null=False, blank=False)

    def __str__(self):
        return self.prediction



