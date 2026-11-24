'''
Quick summary of the dataset: counts per source and class
Run: python -m scripts.explore_dataset
'''
import pandas as pd

from src.data.dataset import get_image_paths_and_labels

def main():
    images, labels = get_image_paths_and_labels()
    df = pd.DataFrame({
        "source": ["A" if "DatasetA" in str(p) else "B" for p in images],
        "class": ["nondefective" if l is None else "defective" for l in labels],
    })

    print(f"Total images: {len(df)}\n")
    print(df.groupby(["source", "class"]).size().unstack())

if __name__ == "__main__":
    main()
