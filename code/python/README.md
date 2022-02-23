# How to run

1. Download samples given by Physionet from
   [here](https://physionet.org/files/challenge-2016/1.0.0/training.zip)
2. Run the [matlab](../matlab/) script to generate the `S1.csv` file
3. It's recommended to create a python virtual environment to install all the
   necessary packages. You can do that by running:
   ```sh
   python -m venv name_of_venv      # to create virtual environment
   source name_of_venv/bin/activate # to activate virtual environment
   pip install -r requirements.txt  # to install all the necessary packages
   ```
4. Edit the `mfcc.py` and `spectogram.py` files to change the path for both the
   samples you downloaded and the `S1.csv`
5. Inside `cnn.py` at the main function you can find
   `get_dataset(use_spectograms=False)`. Run the `cnn.py` file with
   `use_spectograms=False` to train the model using the MFCCs as input or set it
   to `use_spectograms=True` to train the model with spectograms as input.
   ```
   python cnn.py
   ```
