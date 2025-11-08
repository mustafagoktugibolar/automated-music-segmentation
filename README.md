# Music Segmentation Project

This project is a automated music segmentation tool.

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

### Installation

1.  **Clone the repository**

    If you haven't already, clone the project repository to your local machine.
    ```bash
    git clone <https://github.com/mustafagoktugibolar/automated-music-segmentation.git>
    cd music-segmentation
    ```

2.  **Create the Conda Environment**

    Use the `environment.yml` file to create a new Conda environment with all the required dependencies. This will create an environment named `music-segmentation-env`.

    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the Environment**

    To start using the project, you must activate the Conda environment in your terminal session.

    ```bash
    conda activate music-segmentation-env
    ```
    Your terminal prompt should now indicate that you are in the `(music-segmentation-env)`.

## Running the Application

Once the setup is complete and the environment is activated, you can run the backend application.

```bash
python backend/main.py
```

This will execute the main Python script and start the backend service.
