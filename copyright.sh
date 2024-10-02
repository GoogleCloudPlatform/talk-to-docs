#!/bin/bash

# The copyright notice to add to each Python file
COPYRIGHT="# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."

# Loop through all .py files in the current directory and subdirectories
find . -name "*.py" | while read -r file; do
    # Check if the file already contains the copyright (skip if it does)
    if ! grep -q "Copyright 2024 Google LLC" "$file"; then
        # Add the copyright notice to the beginning of the file
        echo "$COPYRIGHT" | cat - "$file" > temp && mv temp "$file"
        echo "Updated: $file"
    else
        echo "Already updated: $file"
    fi
done

echo "All .py files have been processed."
