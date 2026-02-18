To run this project, create test directories in the repo containing images to train/test on. You can then run the script as such:

Activate virtual environment:
```source venv/bin/activate```

Train based on [directory]: ```python classifier2_mlp.py --train```

**Note:** you can use ```--gpu``` to run using a Macbook M1 GPU through Metal, and ```--epochs [number]``` to run [number] training epochs

Classify images: ```python classifier2_mlp.py -d [imgDirectory]```

Classifier 1 is a single perceptron, classifier 2 is a multi-layered perceptron model, and classifier 3 is a convolutional neural network.
