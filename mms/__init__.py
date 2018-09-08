# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
This module does the following:
a. Starts model-server.
b. Creates end-points based on the configured models.
c. Exposes standard "ping" and "api-description" endpoints.
d. Waits for servicing inference requests.
"""
from . import log
from . import version
from . import utils
from . import metrics
from . import model_service
from . import arg_parser

__version__ = version.__version__
