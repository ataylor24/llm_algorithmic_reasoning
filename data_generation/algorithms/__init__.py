# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CLRS algorithm implementations."""

# pylint:disable=g-bad-import-order

from .divide_and_conquer import find_maximum_subarray
from .divide_and_conquer import find_maximum_subarray_kadane

from .dynamic_programming import matrix_chain_order
from .dynamic_programming import lcs_length
from .dynamic_programming import optimal_bst

from .geometry import segments_intersect
from .geometry import graham_scan
from .geometry import jarvis_march

from .graphs import dfs
from .graphs import bfs
from .graphs import topological_sort
from .graphs import articulation_points
from .graphs import bridges
from .graphs import strongly_connected_components
from .graphs import mst_kruskal
from .graphs import mst_prim
from .graphs import bellman_ford
from .graphs import dijkstra
from .graphs import dag_shortest_paths
from .graphs import floyd_warshall
from .graphs import bipartite_matching

from .greedy import activity_selector
from .greedy import task_scheduling

from .searching import minimum
from .searching import binary_search
from .searching import quickselect

from .sorting import insertion_sort
from .sorting import bubble_sort
from .sorting import heapsort
from .sorting import quicksort

from .strings import naive_string_matcher
from .strings import kmp_matcher
