"""
Testing things out
"""
import xenith
import pandas as pd

tsv = xenith.convert_kojak(kojak="data/test.kojak.txt",
                           perc_inter="data/test.perc.inter.txt",
                           perc_intra="data/test.perc.intra.txt",
                           out_file="data/test.tsv",
                           to_pin=False)

df = pd.read_csv(tsv, sep="\t")

print(df.shape)
