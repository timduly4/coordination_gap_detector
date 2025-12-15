"""
Constants and default parameters for ranking algorithms.
"""

# BM25 Parameters
# k1: Controls term frequency saturation
#     Higher values give more weight to term frequency
#     Typical range: 1.2 - 2.0
#     Default: 1.5
BM25_K1 = 1.5

# b: Controls document length normalization
#    0 = no length normalization
#    1 = full length normalization
#    Typical range: 0.5 - 0.8
#    Default: 0.75
BM25_B = 0.75

# Default index name for messages
DEFAULT_INDEX_NAME = "messages"
