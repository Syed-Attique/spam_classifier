import pandas as pd


def load_new_sms_dataset(path):
    df = pd.read_csv(path)

    df.columns = [
        column.strip().lower()
        for column in df.columns
    ]

    df["label"] = df["label"].str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["text"])

    dataset = []

    for _, row in df.iterrows():
        label = row["label"]
        message = row["text"]

        if label == "ham":
            dataset.append((message, "ham"))

        elif label in ["spam", "smishing"]:
            dataset.append((message, "spam"))

        else:
            raise ValueError(f"Unknown label found: {label}")

    return dataset