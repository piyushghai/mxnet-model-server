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

from collections import namedtuple
import pytest
from mock import patch, mock_open

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
            patches.unlink.assert_called_once()
            patches.log.assert_called_once()
            patches.socket.assert_called_once()

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
            patches.encode.side_effect = Exception('teste')
            with pytest.raises(MMSError) as excinfo:
                MXNetModelServiceWorker.create_predict_response(None, self.sample_ret, self.req_id_map, self.empty_invalid_reqs)
            assert excinfo.value.code == err.CODEC_FAIL
            assert excinfo.value.message == "codec failed Exception('teste',)"


        @pytest.mark.parametrize('value', [b'test', 'test', {'test': True}])
        def test_value_types(self, patches, value):
            ret = [value]
            resp = MXNetModelServiceWorker.create_predict_response(None, ret, self.req_id_map, self.empty_invalid_reqs)
            patches.encode.assert_called_once()
            assert set(resp[0].keys()) == {'requestId', 'code', 'value', 'encoding'}

        # @pytest.mark.parametrize('invalid_reqs', [dict(), {'reqId1': 'invalidCode1'}])
        # def test_with_or_without_invalid_reqs(self, patches, invalid_reqs):
        #     resp = MXNetModelServiceWorker.create_predict_response(None, self.sample_ret, self.req_id_map, invalid_reqs)




def test_emit_metrics(mocker):
    metrics = {'test_emit_metrics': True}
    printer = mocker.patch('builtins.print')
    flush = mocker.patch('sys.stdout.flush')
    emit_metrics(metrics)

    printer.assert_called()
    for k in metrics.keys():
        assert k in printer.call_args_list[1][0][0]
