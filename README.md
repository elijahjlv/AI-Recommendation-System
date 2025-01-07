# AI-Recommendation-System

This project implements a basic AI-powered recommendation system using TF-IDF and cosine similarity. It is designed to recommend similar items based on descriptions.

---

## Features

- **Input**: Dataset with item descriptions.
- **Output**: Top N recommended items based on similarity scores.
- **Use Cases**: Finance, e-commerce, and any domain needing content-based recommendations.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:elijahlv/AI-Recommendation-System.git
   cd AI-Recommendation-System
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system**:
   ```bash
   python main.py
   ```

---

## Usage

1. **Prepare your dataset**:
   - Ensure your dataset is in CSV format with a column containing item descriptions.

2. **Configure parameters**:
   - Modify the `config.json` file to set the input file path, output file path, and the number of recommendations (N).

3. **Run the script**:
   ```bash
   python main.py
   ```

4. **Output**:
   - The system will generate a CSV file with recommended items based on the input descriptions.

---

## Example

**Input**: Dataset containing descriptions of books.

**Output**: Top 5 similar books for each book in the dataset.

---

## Technologies Used

- **Python**: Core programming language.
- **Libraries**:
  - `scikit-learn`: For implementing TF-IDF and cosine similarity.
  - `pandas`: For data manipulation.
  - `numpy`: For numerical operations.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

If you have any questions or suggestions, feel free to reach out:
- **GitHub**: [Elijah Johnson](https://github.com/elijahlv)

