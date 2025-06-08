from datasets import load_dataset
import torchvision.transforms as transforms

# Load CIFAR-100 in streaming mode with minimal processing
dataset = load_dataset("cifar100", streaming=True)

# Define basic transform - resize to 224x224 only
transform = transforms.Resize((224, 224))

# Process first 500 samples for quick testing
train_samples = []
test_samples = []
for i, sample in enumerate(dataset["train"]):
    if i >= 500:
        break
    train_samples.append({"image": transform(sample["img"]), "label": sample["fine_label"]})

for i, sample in enumerate(dataset["test"]):
    if i >= 100:
        break
    test_samples.append({"image": transform(sample["img"]), "label": sample["fine_label"]})

print(f"Loaded {len(train_samples)} train and {len(test_samples)} test samples")