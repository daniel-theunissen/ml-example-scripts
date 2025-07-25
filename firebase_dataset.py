import json
import random

class SimpleDataset:
    def __init__(self, anchors, pos_neg, labels):
        self.anchors = list(anchors)  
        self.pos_neg = list(pos_neg)  
        self.labels = list(labels) 

    def select(self, indices):
        """Select a subset of the dataset based on the provided indices."""
        return SimpleDataset(
            [self.anchors[i] for i in indices],
            [self.pos_neg[i] for i in indices],
            [self.labels[i] for i in indices]
        )

class FirebaseDataset:
    def __init__(self, file_path, train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1):
        self.file_path = file_path
        self.train_data = None
        self.test_data = None
        self.eval_data = None

        # Perform the split immediately upon initialization
        self.train_test_eval_split(train_ratio, test_ratio, eval_ratio)

    def load_data(self):
        """Load JSON data from a file."""
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print("Error loading JSON data:", e)
            return {}

    def train_test_eval_split(self, train_ratio, test_ratio, eval_ratio):
        """Load data and split the dataset into train, test, and eval sets."""
        total_ratio = train_ratio + test_ratio + eval_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError("Ratios must sum to 1.0")

        # Load the data
        data = self.load_data()
        
        # Prepare the data for processing
        anchors = []
        pos_neg = []
        labels = []

        for key, value in data.items():
            anchors.append(value['question'])
            pos_neg.append(value['answer'])
            labels.append(value['is_pair'])

        # Combine the data into a list of tuples
        combined_data = list(zip(anchors, pos_neg, labels))
        
        # Shuffle the combined data
        random.shuffle(combined_data)

        # Calculate the split indices
        total_size = len(combined_data)
        train_size = int(total_size * train_ratio)
        test_size = int(total_size * test_ratio)

        # Split the data
        train_data = combined_data[:train_size]
        test_data = combined_data[train_size:train_size + test_size]
        eval_data = combined_data[train_size + test_size:]

        # Unzip the data back into separate lists
        self.train_data = SimpleDataset(*zip(*train_data))
        self.test_data = SimpleDataset(*zip(*test_data))
        self.eval_data = SimpleDataset(*zip(*eval_data))

    def __getitem__(self, key):
        """Allow access to train, test, and eval datasets using dictionary-like syntax."""
        if key == "train":
            return self.train_data
        elif key == "test":
            return self.test_data
        elif key == "eval":
            return self.eval_data
        else:
            raise KeyError(f"Invalid key: {key}")
        
    @property
    def size(self):
        """Return the total number of data points across all datasets."""
        train_count = len(self.train_data.anchors) if self.train_data else 0
        test_count = len(self.test_data.anchors) if self.test_data else 0
        eval_count = len(self.eval_data.anchors) if self.eval_data else 0
        return train_count + test_count + eval_count
