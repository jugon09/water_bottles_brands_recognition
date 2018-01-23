from django.core.files import File
from django.shortcuts import render

from .recognition.cnn_image_generator import test_one_image
from .form import FileForm
from .models import Prediction


def upload_image(request):
    template = 'prediction.html'
    clase = ''
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            with open('aux.jpg', 'wb') as f:
                f.write(request.FILES['file'].read())
                clase = test_one_image(f.name)
            Prediction.objects.create(image=File(request.FILES['file']), prediction=clase)
    else:
        form = FileForm()
    return render(request, template, {'form': form, 'clase': clase})
