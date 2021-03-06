from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic.base import View
from braces.views import JSONResponseMixin
import json
from django.conf import settings
import numpy as np
from os import listdir
from os.path import isfile, join

# Create your views here.


class RlAssistPredictedView(JSONResponseMixin, View):

    def post(self, request, *args, **kwargs):
        """
        :param request: API request
        :param args: API args
        :param kwargs: API kwargs
        :return: save/delete user preference settings based on the user id
        """
        data = json.loads(self.request.body.decode('utf-8'))
        status , result  = self.post_api_response(data)
        return self.render_json_response(dict(status=status,result = result))

    def get(self, request, *args, **kwargs):
        """
        :param request: API request
        :param args: API args
        :param kwargs: API kwargs
        :return: Get worklist user preference settings based on the user and its selected worklist
        """
        data = self.qdict_to_dict(request.GET)
        status , result = self.get_api_response(data)
        return self.render_json_response(dict(status=status,result = result))

    def qdict_to_dict(self, qdict):
        """Convert a Django QueryDict to a Python dict.

        Single-value fields are put in directly, and for multi-value fields, a list
        of all values is stored at the field's key.

        """
        return {k: json.loads(v[0]) if len(v) == 1 else v for k, v in qdict.lists()}

    def post_api_response(self, data):
        """
        :param data: Input json data from GUI
        :return: save/delete input json in db by differentiating using "action_name" flag
        """
        kwargs = {
            'is_active': 1
        }
        return 'success', kwargs

    def get_files(self, path):
        full_path = '{}/{}'.format(settings.MEDIA_ROOT, path)
        files = [f for f in listdir(full_path) if isfile(join(full_path, f))]
        return files

    def get_file_path(self, path):
        full_path = '{}/{}'.format(settings.MEDIA_ROOT, path)
        files = [dict(path=join(settings.MEDIA_URL, path, f), file=f) for f in listdir(full_path) if isfile(join(full_path, f))]
        return files

    def get_api_response(self, data):
        files= self.get_files('rlassist/')
        all_program_dtls= {}
        for file in files:
            full_path = '{}/{}/{}'.format(settings.MEDIA_ROOT, "rlassist", file)
            rlassist_program_dtls= np.load(full_path, allow_pickle=True)
            program_dtls= rlassist_program_dtls.tolist()
            all_program_dtls.update(program_dtls)

        kwargs = {
            'program_dtls': all_program_dtls
        }
        return 'success', kwargs