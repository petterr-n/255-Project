# DAT255 Deep Learning project

## Google Speech Command dataset

The dataset used for training, validating and testing the deep neural network, is formatted like the following:

- file: Path to the audio file.
- audio['array']: The actual audio data as a numpy array.
- audio['sampling_rate']: The sampling rate of the audio file (16KHz).
- label: The label for the audio file (e.g. a specific command like "yes", "no", etc.).
- is_unknown: Boolean flag indicating wether the sample is unknown.
- speaker_id: The speakers ID who recordede the sample.
- utterance_id: The ID of the specific utterance of that speaker. 

---

## Installation and Usage

Follow these steps to set up the project locally:

1. Clone the repository:

    ```
    git clone https://github.com/your-username/your-repository-name.git
    ```

2. Navigate to the project directory:

    ```
    cd your-repository-name
    ```

3. Set up a **virtual environment**:

    ```
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    - On **macOS/Linux**:

        ```
        source venv/bin/activate
        ```

5. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

---

# Usage

To run the project:

1. Activate the virtual environment (if not already active) (Steg 4).
2. Run the main script:

    ```
    python test.py
    ```

---


