# Quantified Self-Portrait - 3D Force Graph Visualization

## Description
This project was developed as part of the **Data Art** module in my master's program. The objective was to create a **self-portrait** by analyzing selected Instagram chats, generating a **Co-Occurrence Matrix**, and visualizing it through a **3D Force Graph**. This visualization provides an artistic representation of my interactions in a unique data-driven form. To view my graph, skip the first steps and run directly the python -m http.server command.

## Project Overview 
This project involves three key components:
1. **Data Pre-processing (dataAnalysis)**: Pre-processing selected Instagram chats to create a **Co-Occurrence Matrix**.
2. **3D Visualization (index.html)**: The matrix data is visualized using a 3D Force Graph, rendered in the `index.html`.
3. **Data (graph_data.json)**: Nodes and Link data generated in dataAnalysis used to generate the graph.


## Installation

### Prerequisites
To run this project locally, you need the following:
- Python 3.x
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)

## Steps for own data Processing

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
    ```

2. **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
3. **Activate the virtual environment:**
    ```bash
    poetry shell
    ```
## Usage

### Running Data Analysis

Adjust the dataAnalysis script to your data, include your instagram message files, adjust your name and run it to create the graph_data.json file. 
   ```bash
    python dataAnalysis.py
   ```
### Viewing the 3D Force Graph 
The index.html file visualizes the generated graph_data.json using a 3D Force Graph. To view the graph run:
   ```bash
    python -m http.server
   ```
Then, navigate to http://localhost:8000/index.html in your browser.
