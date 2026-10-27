# © 2026 Telefónica Innovación Digital S.L.
#
# This file is part of wake-up-word-detection.
#
# wake-up-word-detection is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# wake-up-word-detection is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.
#
# You should have received a copy of the Affero GNU General Public License along with wake-up-word-detection. If not, see https://www.gnu.org/licenses/

import os
import json

class Reporter:
    def __init__(self, experiment_path, filename):
        self.experiment_path = experiment_path
        self.file_name = filename
        self.output_file = os.path.join(experiment_path, self.file_name)
        self.data = {}

    def __del__(self): 
        with open(self.output_file , 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def report(self, key, data):
        self.data[key] = data

    def load(self, filepath):
        self.data = json.load(open(filepath, 'r'))