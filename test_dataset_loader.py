from dataset_loader import load_new_sms_dataset

dataset = load_new_sms_dataset("Dataset_5971.csv")

print("Total:", len(dataset))

ham_count = 0
spam_count = 0

for message, label in dataset:
    if label == "ham":
        ham_count += 1
    else:
        spam_count += 1

print("Ham:", ham_count)
print("Spam/Smishing:", spam_count)

print(dataset[:5])