# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ModelServiceWorker is the worker that is started by the MMS front-end.
Communication message format: JSON message
"""

# pylint: disable=redefined-builtin

import socket
from collections import namedtuple
import pytest
from mock import Mock

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mms.model_service_worker import MXNetModelServiceWorker, emit_metrics

class TestMXNetModelServiceWorker:

    class TestInit:

        socket_name = "sampleSocketName"

        def test_missing_socket_name(self):
            with pytest.raises(ValueError) as excinfo:
                MXNetModelServiceWorker()
            assert excinfo.value.args[0] == 'Incomplete data provided: Model worker expects "socket name"'

        def test_socket_in_use(self, mocker):
            unlink = mocker.patch('os.unlink')
            pathexists = mocker.patch('os.path.exists')
            unlink.side_effect = OSError()
            pathexists.return_value = True

            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker(self.socket_name)
            assert self.socket_name in excinfo.value.message
            assert excinfo.value.code == err.SOCKET_ERROR
            assert excinfo.value.message == 'socket already in use: sampleSocketName.'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['unlink', 'log', 'socket'])
            patches = Patches(
                mocker.patch('os.unlink'),
                mocker.patch('mms.model_service_worker.log_msg'),
                mocker.patch('socket.socket')
            )
            return patches


        @pytest.mark.parametrize('exception', [IOError('testioerror'), OSError('testoserror')])
        def test_socket_init_exception(self, patches, exception):
            patches.socket.side_effect = exception
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker(self.socket_name)
            assert excinfo.value.code == err.SOCKET_ERROR
            assert excinfo.value.message == 'Socket error in init sampleSocketName. {}'.format(repr(exception))

        def test_socket_unknown_exception(self, patches):
            patches.socket.side_effect = Exception('unknownException')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker(self.socket_name)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION
            assert excinfo.value.message == "Exception('unknownException',)"

        def test_success(self, patches):
            MXNetModelServiceWorker(self.socket_name)
            patches.unlink.assert_called_once_with(self.socket_name)
            patches.log.assert_called_once_with('Listening on port: sampleSocketName\n')
            patches.socket.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)

    class TestCreatePredictResponse:
        sample_ret = ['val1']
        req_id_map = {0: 'reqId1'}
        empty_invalid_reqs = dict()
        invalid_reqs = {'reqId1': 'invalidCode1'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['encode'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.ModelWorkerCodecHelper.encode_msg')
            )
            return patches

        def test_codec_exception(self, patches):
            patches.encode.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.create_predict_response(None, self.sample_ret, self.req_id_map, self.empty_invalid_reqs)
            assert excinfo.value.code == err.CODEC_FAIL
            assert excinfo.value.message == "codec failed Exception('testerr',)"


        @pytest.mark.parametrize('value', [(b'test', b'test'), ('test', b'test'), ({'test': True}, b'{"test": true}')])
        def test_value_types(self, patches, value):
            ret = [value[0]]
            resp = MXNetModelServiceWorker.create_predict_response(None, ret, self.req_id_map, self.empty_invalid_reqs)
            patches.encode.assert_called_once_with('base64', value[1])
            assert set(resp[0].keys()) == {'requestId', 'code', 'value', 'encoding'}

        @pytest.mark.parametrize('invalid_reqs,requestId,code,value,encoding', [
            (dict(), 'reqId1', 200, 'encoded', 'base64'),
            ({'reqId1': 'invalidCode1'}, 'reqId1', 'invalidCode1', 'encoded', 'base64')
        ])
        def test_with_or_without_invalid_reqs(self, patches, invalid_reqs, requestId, code, value, encoding):
            patches.encode.return_value = 'encoded'
            resp = MXNetModelServiceWorker.create_predict_response(None, self.sample_ret, self.req_id_map, invalid_reqs)
            assert resp == [{'requestId': requestId, 'code': code, 'value': value, 'encoding': encoding}]

    class TestPredict:
        data = {u'modelName': 'modelName', u'requestBatch': ['data']}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['worker', 'validate', 'emit', 'model_service'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.MXNetModelServiceWorker', spec=['service_manager', 'retrieve_data_for_inference', 'create_predict_response']),
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_predict_msg'),
                mocker.patch('mms.model_service_worker.emit_metrics'),
                Mock(['metrics_init', 'inference', 'metrics_store'])
            )
            patches.worker.service_manager.get_loaded_modelservices.return_value = {'modelName': patches.model_service}
            patches.worker.retrieve_data_for_inference.return_value = [{0: 'inputBatch1'}], 'req_id_map', 'invalid_reqs'
            patches.model_service.metrics_store.store = Mock()
            return patches

        def test_value_error(self, patches):
            patches.validate.side_effect = ValueError('testerr')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.predict(patches.worker, self.data)
            assert excinfo.value.code == err.INVALID_PREDICT_MESSAGE
            assert excinfo.value.message == "ValueError('testerr',)"

        def test_pass_mms_error(self, patches):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            patches.worker.service_manager.get_loaded_modelservices.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.predict(patches.worker, self.data)
            assert excinfo.value == error

        def test_not_loaded(self, patches):
            patches.worker.service_manager.get_loaded_modelservices.return_value = []
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.predict(patches.worker, self.data)
            assert excinfo.value.code == err.MODEL_SERVICE_NOT_LOADED
            assert excinfo.value.message == "Model modelName is currently not loaded"

        def test_invalid_batch_size(self, patches):
            data = {u'modelName': 'modelName', u'requestBatch': ['data1', 'data2']}
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.predict(patches.worker, data)
            assert excinfo.value.code == err.UNSUPPORTED_PREDICT_OPERATION
            assert excinfo.value.message == "Invalid batch size 2"

        def test_success(self, patches):
            response, msg, code = MXNetModelServiceWorker.predict(patches.worker, self.data)
            patches.validate.assert_called_once_with(self.data)
            patches.worker.retrieve_data_for_inference.assert_called_once_with(['data'], patches.model_service)
            patches.model_service.inference.assert_called_once_with(['inputBatch1'])
            patches.emit.assert_called_once_with(patches.model_service.metrics_store.store)
            patches.worker.create_predict_response.assert_called_once_with([patches.model_service.inference()], 'req_id_map', 'invalid_reqs')
            assert response == patches.worker.create_predict_response()
            assert msg == "Prediction success"
            assert code == 200

    class TestLoadModel:

        data = {'modelPath': 'mpath', 'modelName': 'name', 'handler': 'handled'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['worker', 'validate', 'loader'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.MXNetModelServiceWorker'),
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_load_message'),
                mocker.patch('mms.model_service_worker.ModelLoader.load')
            )
            patches.loader.return_value = 'testmanifest', 'test_service_file_path'
            return patches

        def test_load_value_error(self, patches):
            patches.loader.side_effect = ValueError('verror')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.load_model(patches.worker, self.data)
            assert excinfo.value.code == err.VALUE_ERROR_WHILE_LOADING
            assert excinfo.value.message == 'verror'

        def test_pass_mms_error(self, patches):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            patches.loader.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.load_model(patches.worker, self.data)
            assert excinfo.value == error

        def test_unknown_error(self, patches):
            patches.loader.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.load_model(patches.worker, self.data)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION_WHILE_LOADING
            assert excinfo.value.args[0] == "Exception('testerr',)"

        @pytest.mark.parametrize('batch_size', [(None, None), ('1', 1)])
        @pytest.mark.parametrize('gpu', [(None, None), ('2', 2)])
        def test_optional_args(self, patches, batch_size, gpu):
            data = self.data.copy()
            if batch_size[0]:
                data['batchSize'] = batch_size[0]
            if gpu[0]:
                data[u'gpu'] = gpu[0]
            MXNetModelServiceWorker.load_model(patches.worker, data)
            patches.worker.service_manager.register_and_load_modules.assert_called_once_with('name', 'mpath', 'testmanifest', 'test_service_file_path', gpu[1], batch_size[1])

        def test_success(self, patches):
            msg, code = MXNetModelServiceWorker.load_model(patches.worker, self.data)
            patches.validate.assert_called_once_with(self.data)
            patches.loader.assert_called_once_with('mpath', 'handled')
            assert msg == 'loaded model test_service_file_path'
            assert code == 200

    class TestUnloadModel:
        request = {u'model-name': 'mname'}

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['worker', 'validate'])
            patches = Patches(
                mocker.patch('mms.model_service_worker.MXNetModelServiceWorker'),
                mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators.validate_unload_msg')
            )
            return patches

        def test_not_loaded(self, patches):
            patches.worker.service_manager.unload_models.side_effect = KeyError()
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.unload_model(patches.worker, self.request)
            assert excinfo.value.code == err.MODEL_CURRENTLY_NOT_LOADED
            assert excinfo.value.args[0] == 'Model is not being served on model server'

        def test_pass_mms_error(self, patches):
            error = MMSError(err.UNKNOWN_COMMAND, 'testerr')
            patches.worker.service_manager.unload_models.side_effect = error
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.unload_model(patches.worker, self.request)
            assert excinfo.value == error


        def test_unknown_error(self, patches):
            patches.worker.service_manager.unload_models.side_effect = Exception('testerr')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.unload_model(patches.worker, self.request)
            assert excinfo.value.code == err.UNKNOWN_EXCEPTION
            assert excinfo.value.args[0] == "Unknown error Exception('testerr',)"

        def test_success(self, patches):
            msg, code = MXNetModelServiceWorker.unload_model(patches.worker, self.request)
            patches.validate.assert_called_once_with(self.request)
            patches.worker.service_manager.unload_models.assert_called_once_with('mname')
            assert msg == "Unloaded model mname"
            assert code == 200




def test_emit_metrics(mocker):
    metrics = {'test_emit_metrics': True}
    printer = mocker.patch('mms.model_service_worker.print')
    flush = mocker.patch('sys.stdout.flush')
    emit_metrics(metrics)

    printer.assert_called()
    for k in metrics.keys():
        assert k in printer.call_args_list[1][0][0]
