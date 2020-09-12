#!/usr/bin/env python

import pstats
import sys

def main(fn):
    p = pstats.Stats(fn)
    p.strip_dirs().sort_stats('tottime').print_stats(10)

if __name__=='__main__':
    main(sys.argv[1])
