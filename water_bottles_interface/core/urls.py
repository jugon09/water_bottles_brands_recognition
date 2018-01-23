from django.conf.urls import url

from .views import upload_image

app_name = 'core'

urlpatterns = [
    url(r'^$', upload_image, name='home')]