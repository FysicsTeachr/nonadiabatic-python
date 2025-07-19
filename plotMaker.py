import pandas as pd
import matplotlib.pyplot as plt
import re

AU_TO_FS = 24.18884

def load_data(filename):
    """
    Loads only data lines (ignores comments/headers not starting with number).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = []
    for line in lines:
        if re.match(r'^\s*[\d\.\+\-Ee]', line):
            data_lines.append(line)
    if not data_lines:
        raise ValueError(f"No data lines detected in {filename}")
    from io import StringIO
    return pd.read_csv(StringIO(''.join(data_lines)), sep=r'\s+', header=None)

def convert_time_column(df, AU_flag):
    if AU_flag.lower() == 'yes':
        df[0] = df[0] * AU_TO_FS
    return df

def plot_data(file1, AU1, p1, 
              file2=None, AU2=None, p2=None):
    plt.figure(figsize=(10,6))
    
    # Load and plot first file
    if file1:
        data1 = load_data(file1)
        data1 = convert_time_column(data1, AU1)
        time1 = data1[0]
        print(f"File1 '{file1}' columns available: {list(range(1, data1.shape[1]+1))}")
        for idx in p1:
            col = idx - 1
            if col >= data1.shape[1]:
                raise ValueError(f"Column {idx} is out of bounds for file1")
            plt.plot(time1, data1[col], label=f'{file1}: col {idx}')
    
    # If second dataset is specified
    if file2 and p2:
        data2 = load_data(file2)
        data2 = convert_time_column(data2, AU2)
        time2 = data2[0]
        print(f"File2 '{file2}' columns available: {list(range(1, data2.shape[1]+1))}")
        for idx in p2:
            col = idx - 1
            if col >= data2.shape[1]:
                raise ValueError(f"Column {idx} is out of bounds for file2")
            plt.plot(time2, data2[col], linestyle='--', label=f'{file2}: col {idx}')
    
    plt.xlabel('Time (fs)' if AU1.lower() == 'yes' or (AU2 and AU2.lower() == 'yes') else 'Time (given units)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.title('Custom Plot')
    plt.show()


