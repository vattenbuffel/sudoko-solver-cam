# sudoko-solver-cam
Solves sudoko which is read through cam

![Image](./img/solver.gif "icon")

# How it works
First the sudoku board is isolated. Then the cells are extracted and the number in each cell is identified. The blank cells are identifed by 99 % of the pixels being white. The non black cells are identified via a neural network.

# Requirements
* Python 
* Pip
* A webcam

# How to use
1. Make sure you fulfill the requirements and installed everything.
3. Run ``` python main.py ```

# How to install
Run these commands:
* pip3 install opencv-python && pip3 install numpy && pip3 install torch && pip3 install torchvision 


