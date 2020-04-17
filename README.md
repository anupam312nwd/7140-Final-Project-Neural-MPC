# 7140-Final-Project-Neural-MPC
Final project doing MPC inspired by Neural ODEs

Link to the [Final Project Presenation](https://docs.google.com/presentation/d/1ZqnLlKaLKURLaXhvny5aG1RFxwvfNMs3hhryQak38MM/edit#slide=id.g733d2cba99_8_6)

In order to run the code please run the following:

    pip install torch torchvision torchdiffeq

    python run_gru_agent.py
    python run_node_agent.py

The two python files in the repo will train the corresponding agent if no model file is found, and will run a round of MPC with the trained model. All plots and gifs will be placed in the plots directory.

Agent implementations are in the agents directory. The MPC implementation is in the utils directory.