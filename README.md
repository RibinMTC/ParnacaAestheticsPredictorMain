## ParnacaAestheticsPredictorMain

ParnacaAestheticsPredictorMain is a python tool to predicts meaning image attributes for the aesthetic assessment project. The project utilizes the [Deep Image Aesthetic Analysis](https://github.com/aimerykong/deepImageAestheticsAnalysis) implementation.


1. The predictor uses Flask (by default listening on *localhost:5005*) to listen to incoming requests, which should contain the path to the image to detect the image attributes.
2. The predictor returns the following attributes : *Interesting Content, Object Emphasis, Good Lighting, Color Harmony, Vivid Color, Shallow Depth Of Field, Motion Blur, Rule of Thirds, Balancing Element, Repetition, Symmetry*. 
The attributes are in the range [-1,1], where -1 affects the final score negatively and 1 affects the score positively.

### Requirements

This project uses Python 2 and Anaconda.

### Installation

Clone this repository and create conda environment:

```bash
conda env create -f environment.yml
```

Download the pretrained model resources from [here](https://drive.google.com/file/d/1Xn5ord4ohWnwis1J7NrI2qkw9gEUMOxx/view?usp=sharing)
and extract it into the root directory of the project. If the model resources are saved elsewhere, adjust the path to the resources in the *src/parnaca_prediction/parnaca_aadb_model.py* file in the *model_resources_root_path_str* variable.

### Usage

The recommended usage is with a docker-compose file.

For testing the code without using Flask, run the following code:

```python
from src.parnaca_prediction.parnaca_aadb_model import ParnacaAADBModel

parnaca_aadb_model = ParnacaAADBModel()
parnaca_aadb_model.predict("path/to/image")
```

To start the Flask Api, which listens for incoming prediction requests run:

```bash
gunicorn --config gunicorn_config.py --env API_CONFIG=api_config.json aesthetics_predictor_api_pkg.predictor_api_server:app
```