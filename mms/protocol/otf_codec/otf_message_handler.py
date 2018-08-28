import struct
import time
import json
import sys
from mms.utils.validators.model_worker_message_validator import ModelWorkerOtfMessageValidator as validate
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes

int_size = 4
END_OF_LIST = -2
START_OF_LIST = -1


class OtfCodecHandler:
    def __retrieve_load_msg(self, data):
        """
        MSG Frame Format:
        "
        | 1.0 | int cmd_length | cmd value | int model-name length | model-name value |
        | int model-path length | model-path value |
        | int batch-size length | batch-size value | int handler length | handler value |
        | int gpu id length | gpu ID value |
        "
        :param data:
        :return:
        """
        startTime = time.time()
        msg = dict()
        offset = 0
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size
        msg['modelName'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size
        msg['modelPath'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length
        msg['batchSize'] = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size
        length = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size
        msg['handler'] = struct.unpack('!{}s'.format(length), data[offset:offset+length])[0].decode()
        offset += length
        gpu_id = struct.unpack('!i', data[offset: offset+int_size])[0]
        if gpu_id > 0:
            msg['gpu'] = gpu_id

        print("Time taken to retrieve load message is {} ms".format((time.time() - startTime) * 1000))
        sys.stdout.flush()
        return "load", msg

    def __retrieve_model_inputs(self, data, msg, content_type):
        offset = 0
        end = False
        while end is False:
            model_input = dict()
            length = struct.unpack('!i', data[offset:offset+int_size])[0]
            offset += int_size
            if length > 0:
                model_input['name'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length
            elif length == END_OF_LIST:
                end = True
                continue

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size

            if length > 0:
                model_input['contentType'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size

            if length > 0:
                if ("contentType" in model_input and "json" in model_input['contentType'].lower()) or \
                   ("json" in content_type.lower()):
                    model_input['value'] = struct.unpack('!{}s'.format(length), data[offset:offset+length])[0].decode()
                elif ("contentType" in model_input and "jpeg" in model_input['contentType'].lower()) or \
                     ("jpeg" in content_type.lower()):
                    model_input['value'] = data[offset: offset+length]
                else:
                    raise MMSError(ModelServerErrorCodes.UNKNOWN_CONTENT_TYPE, "Unknown contentType given for the data")
                offset += length
            msg.append(model_input)
        return offset

    def __retrieve_request_batch(self, data, msg):
        offset = 0
        end = False
        while end is False:
            reqBatch = dict()
            length = struct.unpack('!i', data[offset:offset+int_size])[0]
            offset += int_size
            if length > 0:
                reqBatch['requestId'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length
            elif length == END_OF_LIST:
                end = True
                continue

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size
            if length > 0:
                reqBatch['contentType'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
                offset += length

            length = struct.unpack('!i', data[offset: offset+int_size])[0]
            offset += int_size
            if length == START_OF_LIST:  # Beginning of list
                reqBatch['modelInputs'] = list()
                offset += self.__retrieve_model_inputs(data[offset:], reqBatch['modelInputs'], reqBatch['contentType'])

            msg.append(reqBatch)

    def __retrieve_inference_msg(self, data):
        msg = dict()
        startTime = time.time()
        offset = 0
        length = struct.unpack('!i', data[offset:offset+int_size])[0]
        offset += int_size

        if length > 0:
            msg['modelName'] = struct.unpack('!{}s'.format(length), data[offset: offset+length])[0].decode()
        offset += length

        length = struct.unpack('!i', data[offset: offset+int_size])[0]
        offset += int_size

        if length == START_OF_LIST:
            msg['requestBatch'] = list()
            self.__retrieve_request_batch(data[offset:], msg['requestBatch'])

        print("Time taken to retrieve inference message is {} ms".format((time.time() - startTime) * 1000))
        sys.stdout.flush()
        return "predict", msg

    def retrieve_msg(self, data):
        # Validate its beginning of a message
        if validate.validate_message(data=data) is False:
            return MMSError(ModelServerErrorCodes.INVALID_MESSAGE, "Invalid message received")

        cmd = struct.unpack('!i', data[8:12])[0]

        if cmd == 0x01:
            return self.__retrieve_load_msg(data[12:])
        elif cmd == 0x02:
            return self.__retrieve_inference_msg(data[12:])
        else:
            return "unknown", "Wrong command "

    def __encode_inference_response(self, kwargs):
        try:
            startTime = time.time()
            req_id_map = kwargs['req_id_map']
            invalid_reqs = kwargs['invalid_reqs']
            ret = kwargs['resp']
            msg = bytearray()
            msg += struct.pack('!i', -1)  # start of list
            encoding = ""
            for idx, val in enumerate(ret):
                msg += struct.pack("!i", len(req_id_map[idx]))
                msg += struct.pack('!{}s'.format(len(req_id_map[idx])), req_id_map[idx].encode())

                msg += struct.pack('!i', 200)

                if isinstance(val, str):
                    encoding = "text"
                    msg += struct.pack('!i', len(encoding))
                    msg += struct.pack('!{}s'.format(len(encoding)), encoding.encode())

                    msg += struct.pack('!i', len(val))
                    msg += struct.pack('!{}s'.format(len(val)), val.encode())
                elif isinstance(val, bytes):
                    encoding = "binary"
                    msg += struct.pack('!i', len(encoding))
                    msg += struct.pack('!{}s'.format(len(encoding)), encoding.encode())

                    msg += struct.pack('!i', len(val))
                    msg += val
                else:
                    encoding = 'json'
                    json_value = json.dumps(val)
                    msg += struct.pack('!i', len(encoding))
                    msg += struct.pack('!{}s'.format(len(encoding)), encoding.encode())

                    msg += struct.pack('!i', len(json_value))
                    msg += struct.pack('!{}s'.format(len(json_value)), json_value.encode())

            for req in invalid_reqs.keys():
                msg += struct.pack('!i', len(req))
                msg += struct.pack('!{}s'.format(len(req)), req.encode())

                msg += struct.pack('!i', 4)
                msg += struct.pack('!i', invalid_reqs.get(req))
                msg += struct.pack('!i', len("Invalid input provided".encode()))
                msg += struct.pack('!{}s'.format(len("Invalid input provided".encode())),
                                   "Invalid input provided".encode())
                encoding = "text"
                msg += struct.pack('!i', len(encoding))
                msg += struct.pack('!{}s'.format(len(encoding)), encoding.encode())
            msg += struct.pack('!i', -2) # End of list

            print("Time taken for encoding response is {} ms".format((time.time() - startTime) * 1000))
            sys.stdout.flush()
            return msg

        except Exception:
            # TODO: Return a invalid response
            raise

    def __encode_response(self, kwargs):
        msg = bytearray()
        try:
            startTime = time.time()
            msg += struct.pack('!d', 1.0)
            msg += struct.pack('!i', int(kwargs['code']))
            msg_len = len(kwargs['message'])
            msg += struct.pack('!i', msg_len)
            msg += struct.pack('!{}s'.format(msg_len), kwargs['message'].encode())
            if 'predictions' in kwargs and kwargs['predictions'] is not None:
                msg += kwargs['predictions']
            else:
                msg += struct.pack('!i', 0)  # no predictions
            msg += bytes("\r\n".encode())
            print("Time taken to encode general response is {} ms".format((time.time() - startTime) * 1000))
            sys.stdout.flush()
        except Exception:
            raise
        return msg

    def create_response(self, cmd, **kwargs):
        if cmd == 2: # Predict request response
            return self.__encode_inference_response(kwargs=kwargs)
        if cmd == 3: # All responses
            return self.__encode_response(kwargs=kwargs)
