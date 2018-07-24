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

model_service_worker = MXNetModelServiceWorker('my-socket')

@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket', 'log_msg'])
    mock_patch = Patches(mocker.patch('socket.socket'), mocker.patch('mms.model_service_worker.log_msg'))

    mock_patch.socket.recv.return_value = "{}\r\n"
    return mock_patch


@pytest.fixture()
def json_patches(mocker):
    Patches = namedtuple('Patches', ['json_load'])
    mock_patch = Patches(mocker.patch('json.loads'))
    return mock_patch


def test_recv_msg_with_nil_pkt(socket_patches):
    socket_patches.socket.recv.return_value = None
    with pytest.raises(SystemExit) as ex:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert ex.value.args[0] == 1  # The exit status is exit(1)


def test_rcv_msg_with_IO_Error(socket_patches):
    socket_patches.socket.recv.side_effect = IOError('IOError')

    with pytest.raises(MMSError) as ex:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert ex.value.get_code() == ModelServerErrorCodes.RECEIVE_ERROR
    assert ex.value.get_message() == "IOError"


def test_rcv_msg_with_OSError(socket_patches):
    socket_patches.socket.recv.side_effect = OSError('OSError')

    with pytest.raises(MMSError) as ex:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert ex.value.get_code() == ModelServerErrorCodes.RECEIVE_ERROR
    assert ex.value.get_message() == "OSError"


def test_rcv_msg_with_Exception(socket_patches):
    socket_patches.socket.recv.side_effect = Exception('Some Exception')

    with pytest.raises(MMSError) as ex:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert ex.value.get_code() == ModelServerErrorCodes.UNKNOWN_EXCEPTION
    assert ex.value.get_message() == "Some Exception"


def test_rcv_msg_with_json_value_error(socket_patches, json_patches):
    err_msg = "Some random json error"
    json_patches.json_load.side_effect = ValueError(err_msg)

    with pytest.raises(MMSError) as ex:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert ex.value.get_code() == ModelServerErrorCodes.INVALID_MESSAGE
    assert ex.value.get_message() == "JSON message format error: {}".format(err_msg)


def test_rcv_msg_with_missing_command(socket_patches, json_patches):
    json_patches.json_load.return_value = {}

    with pytest.raises(MMSError) as err:
        MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    assert err.value.get_code() == ModelServerErrorCodes.INVALID_COMMAND
    assert err.value.get_message() == "Invalid message received"


def test_rcv_msg_return_value(socket_patches, json_patches):
    recv_pkt = {'command': {'Some command'}}
    json_patches.json_load.return_value = recv_pkt

    cmd, data = MXNetModelServiceWorker.recv_msg(socket_patches.socket)

    json_patches.json_load.assert_called()
    assert 'command' in data.keys()
    assert cmd == recv_pkt['command']
    assert data == recv_pkt


def test_send_response_with_IO_Error(socket_patches):
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
            socket_patches.log_msg.assert_called_with(log_call_param)

    # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
    assert ex.value.args[0] == ModelServerErrorCodes.SEND_FAILS_EXCEEDS_LIMITS


def test_send_response_with_OS_Error(socket_patches):
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
            socket_patches.log_msg.assert_called_with(log_call_param)

    # The exit status is exit(SEND_FAILS_EXCEEDS_LIMITS)
    assert ex.value.args[0] == ModelServerErrorCodes.SEND_FAILS_EXCEEDS_LIMITS
