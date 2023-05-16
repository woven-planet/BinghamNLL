# Copyright 2023 TOYOTA MOTOR CORPORATION

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import glob
from os.path import basename, splitext
from setuptools import setup

src_dir = "bingham-rotation-learning"
setup(
    name="brotlearn",
    version="1.0.0",
    package_dir={"": src_dir},
    py_modules=[splitext(basename(fname))[0] for fname in glob.glob("{}/*.py".format(src_dir))],
)
