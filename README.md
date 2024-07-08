# Tristar Test

This project is for the Tristar test. This README file provides instructions on how to set up the project environment and run the model_script.

## Setup

1. Clone the repository:

    ```shell
    git clone https://github.com/araghuram3/tristar_test.git
    ```

2. Navigate to the project directory:

    ```shell
    cd tristar_test
    ```

3. Create a virtual environment:

    ```shell
    python -m venv venv
    ```

4. Activate the virtual environment:

    - For Windows:

      ```shell
      venv\Scripts\activate
      ```

    - For macOS/Linux (untested):

      ```shell
      source venv/bin/activate
      ```

5. Install project dependencies from the requirements.txt file:

    ```shell
    pip install -r requirements.txt
    ```

    This will install torch and torchvision cpu versions.
    I installed the GPU versions using:
    
    ```shell
    pip install torch torchvision -f https://download.pytorch.org/whl/cu121/torch_stable.html
    ```
    because I have a CUDA 12.1 on my machine.

6. Add data folder to the project directory and add the data files to the data folder.
Structure should look like this:
    ```shell
    your-repo/
    ├── data/
    │   ├── train/
    │   │   ├── Benign/
    │   │   └── Malignant/
    │   └── test/
    ├── model_script.py
    ├── models.py
    ├── README.md
    └── requirements.txt
    ```
## Usage

To run the model_script, execute the following command:
```shell
python model_script.py
```
This will train the model. 
Different models can be used by selecting a differnet model in model_script.py.
These models are defined in models.py.

The trained model will be saved to the models folder.
The one that comes in this repo was trained with the VGG16 model (defined in models.py) and has an accuracy of 0.93.