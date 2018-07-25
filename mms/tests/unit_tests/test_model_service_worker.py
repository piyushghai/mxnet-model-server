# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest
from collections import namedtuple
from mms.model_service_worker import MXNetModelServiceWorker, MAX_FAILURE_THRESHOLD
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes
from mock import patch
import json
import socket


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket', 'log_msg', 'msg_validator', 'codec_helper', 'json_load', 'log_error'])
    mock_patch = Patches(mocker.patch('socket.socket'), mocker.patch('mms.model_service_worker.log_msg'),
                         mocker.patch('mms.model_service_worker.ModelWorkerMessageValidators'),
                         mocker.patch('mms.model_service_worker.ModelWorkerCodecHelper'),
                         mocker.patch('json.loads'), mocker.patch('mms.model_service_worker.log_error'))

    mock_patch.socket.recv.return_value = "{}\r\n"
    return mock_patch


@pytest.fixture()
def model_service_worker(socket_patches):
    model_service_worker = MXNetModelServiceWorker('my-socket')
    model_service_worker.sock = socket_patches.socket

    return model_service_worker


def test_retrieve_model_input(socket_patches, model_service_worker):
    valid_inputs = [{'encoding': 'base64', 'value': 'val1', 'name': 'model_input_name'}]

    socket_patches.codec_helper.decode_msg.return_value = "some_decoded_resp"

    expected_response = {'model_input_name': 'some_decoded_resp'}

    model_in = model_service_worker.retrieve_model_input(valid_inputs)

    socket_patches.msg_validator.validate_predict_inputs.assert_called()
    socket_patches.codec_helper.decode_msg.assert_called()

    assert expected_response == model_in


class TestCreateAndSendResponse():
    message = 'hello socket'
    code = 007
    resp = {'code' : code, 'message' : message}

    @pytest.fixture()
    def get_send_response_spy(self, model_service_worker, mocker):
        return mocker.patch.object(model_service_worker, 'send_response', wraps=model_service_worker.send_response)

    def test_with_preds(self, socket_patches, model_service_worker, get_send_response_spy):

        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message)
        get_send_response_spy.assert_called_with(socket_patches.socket, json.dumps(self.resp))

        preds = "some preds"
        self.resp['predictions'] = preds
        model_service_worker.create_and_send_response(socket_patches.socket, self.code, self.message, preds)
        get_send_response_spy.assert_called_with(socket_patches.socket, json.dumps(self.resp))

        del(self.resp['predictions'])

    def test_with_exception(self, socket_patches, model_service_worker, get_send_response_spy):
        message = 'hello socket'
        code = 007
        resp = {'code': code, 'message': message}

        get_send_response_spy.side_effect = Exception('Some Exception')
        with pytest.raises(Exception):
            model_service_worker.create_and_send_response(socket_patches.socket, code, message)

        socket_patches.log_error.assert_called()


class TestRecvMsg():
    def test_with_nil_pkt(self, socket_patches):
        socket_patches.socket.recv.return_value = None
        with pytest.raises(SystemExit) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.args[0] == 1  # The exit status is exit(1)

    def test_with_IO_Error(self, socket_patches):
        socket_patches.socket.recv.side_effect = IOError('IOError')

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == ModelServerErrorCodes.RECEIVE_ERROR
        assert ex.value.get_message() == "IOError"

    def test_with_OSError(self, socket_patches):
        socket_patches.socket.recv.side_effect = OSError('OSError')

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == ModelServerErrorCodes.RECEIVE_ERROR
        assert ex.value.get_message() == "OSError"

    def test_with_Exception(self, socket_patches):
        socket_patches.socket.recv.side_effect = Exception('Some Exception')

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == ModelServerErrorCodes.UNKNOWN_EXCEPTION
        assert ex.value.get_message() == "Some Exception"

    def test_with_json_value_error(self, socket_patches):
        err_msg = "Some random json error"
        socket_patches.json_load.side_effect = ValueError(err_msg)

        with pytest.raises(MMSError) as ex:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert ex.value.get_code() == ModelServerErrorCodes.INVALID_REQUEST
        assert ex.value.get_message() == "JSON message format error: {}".format(err_msg)

    def test_with_missing_command(self, socket_patches):
        socket_patches.json_load.return_value = {}

        with pytest.raises(MMSError) as err:
            MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        assert err.value.get_code() == ModelServerErrorCodes.INVALID_COMMAND
        assert err.value.get_message() == "Invalid message received"

    def test_return_value(self, socket_patches):
        recv_pkt = {'command': {'Some command'}}
        socket_patches.json_load.return_value = recv_pkt

        cmd, data = MXNetModelServiceWorker.recv_msg(socket_patches.socket)

        socket_patches.json_load.assert_called()
        assert 'command' in data.keys()
        assert cmd == recv_pkt['command']
        assert data == recv_pkt


class TestSendResponse():

    def test_with_io_error(self, socket_patches, model_service_worker):
        io_error = IOError("IO Error")
        socket_patches.socket.send.side_effect = io_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(ModelServerErrorCodes.SEND_MSG_FAIL, repr(io_error), ''.join([msg, '\r\n']))

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):

                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == ModelServerErrorCodes.SEND_FAILS_EXCEEDS_LIMITS

    def test_with_os_error(self, socket_patches, model_service_worker):
        os_error = OSError("OS Error")
        socket_patches.socket.send.side_effect = os_error
        msg = 'hello socket'

        log_call_param = "{}: Send failed. {}.\nMsg: {}".format(ModelServerErrorCodes.SEND_MSG_FAIL, repr(os_error), ''.join([msg, '\r\n']))

        model_service_worker.send_failures = 0

        with pytest.raises(SystemExit) as ex:
            for i in range(1, MAX_FAILURE_THRESHOLD + 1):

                model_service_worker.send_response(socket_patches.socket, msg)
                socket_patches.socket.send.assert_called()
                assert model_service_worker.send_failures == i
                socket_patches.log_error.assert_called_with(log_call_param)

        # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
        assert ex.value.args[0] == ModelServerErrorCodes.SEND_FAILS_EXCEEDS_LIMITS


class TestStopServer():
    def test_with_nil_sock(self, model_service_worker):
        with pytest.raises(ValueError) as error:
            model_service_worker.stop_server(sock=None)

        assert isinstance(error.value, ValueError)
        assert error.value.args[0] == "Invalid parameter passed to stop server connection"

    def test_with_exception(self, socket_patches, model_service_worker):
        close_exception = Exception("exception")
        socket_patches.socket.close.side_effect = close_exception
        log_call_param = "Error closing the socket {}. Msg: {}".format(socket_patches.socket, repr(close_exception))

        model_service_worker.stop_server(sock=socket_patches.socket)

        socket_patches.socket.close.assert_called()
        socket_patches.log_msg.assert_called_with(log_call_param)

    def test_stop_server(self, socket_patches, model_service_worker):
        with patch.object(model_service_worker, 'send_response', wraps=model_service_worker.send_response) as spy:
            model_service_worker.stop_server(socket_patches.socket)
            spy.assert_called()
            socket_patches.socket.close.assert_called()

class TestRunServer():

    def test_with_socket_bind_error(self, socket_patches, model_service_worker):
        bind_exception = socket.error("binding error")
        socket_patches.socket.bind.side_effect = bind_exception
        with pytest.raises(MMSError) as err:
            model_service_worker.run_server()

        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_not_called()
        assert err.value.get_code() == ModelServerErrorCodes.SOCKET_BIND_ERROR

    def test_with_exception(self, socket_patches, model_service_worker):
        exception = Exception("Some Exception")
        socket_patches.socket.accept.side_effect = exception

        with pytest.raises(Exception) as err:
            model_service_worker.run_server()
        socket_patches.socket.bind.assert_called()
        socket_patches.socket.listen.assert_called()
        socket_patches.socket.accept.assert_called()
        socket_patches.log_msg.assert_called_with("Waiting for a connection")