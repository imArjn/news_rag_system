import pandas as pd

def create_sample(input_file_path, output_file_path, sample_size=1000):
    """
    Reads the full JSON dataset and writes the first sample_size rows to a new JSON file.
    """
    try:
        # Load the full dataset 
        data = pd.read_json(input_file_path, lines=True)
        print(f"Full dataset loaded. Shape: {data.shape}")

        # Select the first sample_size rows
        sample_data = data.head(sample_size)
        print(f"Sample created. Shape: {sample_data.shape}")

        # Save the sample data to a new JSON file in line-delimited format
        sample_data.to_json(output_file_path, orient='records', lines=True)
        print(f"Sample saved to {output_file_path}")
    except Exception as e:
        print("Error creating sample:", e)

if __name__ == '__main__':
    full_dataset_path = "data/Dataset.json"
    sample_output_path = "data/sample.json"
    create_sample(full_dataset_path, sample_output_path, sample_size=1000)
