from django.shortcuts import render
from django.views.generic import TemplateView
from django.conf import settings
from os import listdir
from os.path import isfile, join



class BaseView(TemplateView):
    template_name = "base.html"
    def get_context_data(self, *args, **kwargs):
        context = super(BaseView, self).get_context_data(*args, **kwargs)
        return context

    def get_files(self, path):
        full_path = '{}/{}'.format(settings.MEDIA_ROOT, path)
        files = [f for f in listdir(full_path) if isfile(join(full_path, f))]
        return files

    def get_file_path(self, path):
        full_path = '{}/{}'.format(settings.MEDIA_ROOT, path)
        files = [dict(path=join(settings.MEDIA_URL, path, f), file=f) for f in listdir(full_path) if isfile(join(full_path, f))]
        return files
