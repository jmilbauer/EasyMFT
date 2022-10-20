import csv
from pathlib import Path
import sys

from EasyMFT.analysis import analyze
import EasyMFT.config as cfg
from EasyMFT.mfd import MoralFoundationsDictionary_STRICT

def main(csv_path):
    csv_path = Path(csv_path)
    name = csv_path.stem
    
    mfd_path = (cfg.mfd_path)
    mfd = MoralFoundationsDictionary_STRICT(mfd_path)
    mfd.initialize()
    
    with open(csv_path) as fp:
        reader = csv.DictReader(fp)
        df = analyze(reader, mfd)
        df.to_csv(cfg.analysis_dir / f"{name}.csv")
            
if __name__ == "__main__":
    main(sys.argv[0])