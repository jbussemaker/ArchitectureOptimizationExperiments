"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright: (c) 2020, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
"""

import pytest
import tempfile
from arch_opt_exp.experimenter import Experimenter


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_folder:
        yield tmp_folder


@pytest.fixture(autouse=True)
def experimenter_results_folder(temp_dir):
    Experimenter.results_folder = temp_dir
    yield
    Experimenter.results_folder = None
