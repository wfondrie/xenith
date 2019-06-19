"""
Scratch space for testing things...
"""
import numpy as np
import pandas as pd
import xenith

def main():
    lm = xenith.from_percolator("data/weights.txt")
    res = lm.predict("data/test.tsv")

    return res.estimate_qvalues(metric="xenith_score")

if __name__ == "__main__":
    x = main()
