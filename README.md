Python3 version 3.6.1 or greater is REQUIRED

View our script at:
https://github.com/jguzman-tech/TensorFlowNN/blob/master/main.py

View our plot at:
https://github.com/jguzman-tech/TensorFlowNN/blob/master/300_300_300.png
** 1. Environment setup
These steps assume a linux system.
*** 1.1. First clone the repo 
git clone git@github.com:jguzman-tech/TensorFlowNN.git
*** 1.2. Create an arbitrary directory
#+BEGIN_SRC
mkdir ./.venv
#+END_SRC
*** 1.3. Create a virtual environment
#+BEGIN_SRC
python3 -m venv ./.venv
#+END_SRC
*** 1.4. Activate the virtual environment
#+BEGIN_SRC
. ./.venv/bin/activate
#+END_SRC
**** 1.4.1 You can use an easy to remember bash alias to activate the venv
***** 1.4.1.1 Open your ~/.bashrc
#+BEGIN_SRC
vim ~/.bashrc
#+END_SRC
***** 1.4.1.2 Add this line to your ~/.bashrc
#+BEGIN_SRC
alias act=". ./.venv/bin/activate"
#+END_SRC
You can replace act with any valid bash identifier you want.
***** 1.4.1.3 Now reload your ~/.bashrc
#+BEGIN_SRC
source ~/.bashrc
#+END_SRC
***** 1.4.1.4 Now you can use the "act" command
#+BEGIN_SRC
act
#+END_SRC
This will be saved on all future shells sessions for your current user.
*** 1.5. Install module requirements
#+BEGIN_SRC
pip3 install -r ./requirements.txt
#+END_SRC
** 2. Execution
To see our help message execute:
python3 main.py -h
#+BEGIN_SRC
usage: main.py [-h] max_epochs_one max_epochs_two max_epochs_three

Create three NN models with TensorFlow

positional arguments:
  max_epochs_one    maximum epochs for the 10 hidden units model
  max_epochs_two    maximum epochs for the 100 hidden units model
  max_epochs_three  maximum epochs for the 1000 hidden units model

optional arguments:
  -h, --help        show this help message and exit
#+END_SRC
** 3. Reproduce our results
#+BEGIN_SRC
python3 main.py 300 300 300
#+END_SRC
