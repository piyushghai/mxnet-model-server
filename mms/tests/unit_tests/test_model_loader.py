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
import os
import shutil
import json
from mms.model_loader import MANIFEST_FILENAME
from mms.model_loader import ModelLoader

model_dir_path = 'my-model'
handler_file = 'handler'

manifest_invalid_data_missing_paramsFile = {"model": {}, "engine": {"engineName": "MxNet"}}
manifest_invalid_data_missing_symbolFile = {"model": {"parametersFile": "my-model/params1"},
                                            "engine": {"engineName": "MxNet"}}
manifest_valid_data = {"model": {"parametersFile": "my-model/params1", "symbolFile": 'my-model/symbol.json'},
                       "engine": {"engineName": "MxNet"}}


def empty_file(filepath):

    """
    Utility function for creating an empty file, which will be used throughout these unit tests
    :param filepath:
    :return:
    """

    open(filepath, 'a').close()


@pytest.fixture(scope='session')
def create_empty_manifest_file():
    """
    By setting the scope of create_empty_manifest_file as session we are ensuring that this piece of code runs once for all of the tests combined in this file
    """
    # Setup
    path = '{}/'.format(model_dir_path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    path = '{}/'.format(path) + MANIFEST_FILENAME
    empty_file(path)

    yield path

    # Teardown
    if os.path.exists(path):
        os.remove(path)
        shutil.rmtree('{}/'.format(model_dir_path))


@pytest.mark.usefixtures('create_empty_manifest_file')
def test_manifest_file_open():
    with pytest.raises(Exception):
        ModelLoader.load(model_dir_path, handler_file)


@pytest.mark.usefixtures('create_empty_manifest_file')
def test_parameter_file_defined_in_manifest():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_invalid_data_missing_paramsFile, f)

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, handler_file)

    assert error.value.args[0] == 'parameterFile not defined in MANIFEST.json.'


@pytest.mark.usefixture('create_empty_manifest_file')
def test_parameter_file_exists():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_valid_data, f)

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, handler_file)

    assert error.value.args[0] == 'parameterFile not found: {}.'.format(manifest_valid_data['model']['parametersFile'])


@pytest.mark.usefixture('create_empty_manifest_file')
def test_symbol_file_defined_in_manifest():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_invalid_data_missing_symbolFile, f)

    empty_file(manifest_valid_data['model']['parametersFile'])

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, handler_file)

    assert error.value.args[0] == 'symbolFile not defined in MANIFEST.json.'


@pytest.mark.usefixture('create_empty_manifest_file')
def test_symbol_file_exists():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_valid_data, f)

    empty_file(manifest_valid_data['model']['parametersFile'])

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, handler_file)

    assert error.value.args[0] == 'symbolFile not found: {}.'.format(manifest_valid_data['model']['symbolFile'])


@pytest.mark.usefixture('create_empty_manifest_file')
def test_handler_is_none():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_valid_data, f)

    empty_file(manifest_valid_data['model']['parametersFile'])
    empty_file(manifest_valid_data['model']['symbolFile'])

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, None)

    assert error.value.args[0] == 'No handler is provided.'


@pytest.mark.usefixture('create_empty_manifest_file')
def test_handler_file_not_exists():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_valid_data, f)

    empty_file(manifest_valid_data['model']['parametersFile'])
    empty_file(manifest_valid_data['model']['symbolFile'])

    handler_file_path = os.path.join(model_dir_path, handler_file)

    with pytest.raises(Exception) as error:
        ModelLoader.load(model_dir_path, handler_file)

    assert error.value.args[0] == 'handler file not not found: {}.'.format(handler_file_path)


@pytest.mark.usefixture('create_empty_manifest_file')
def test_return_values_manifest_handler_file():
    with open(model_dir_path + '/' + MANIFEST_FILENAME, 'w') as f:
        json.dump(manifest_valid_data, f)

    empty_file(manifest_valid_data['model']['parametersFile'])
    empty_file(manifest_valid_data['model']['symbolFile'])
    empty_file(os.path.join(model_dir_path, handler_file))

    manifest_return, handler_file_return = ModelLoader.load(model_dir_path, handler_file)
    assert manifest_return == manifest_valid_data
    assert str(handler_file_return) == str(os.path.join(model_dir_path, handler_file))
