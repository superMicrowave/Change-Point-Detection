* NNetOneSplit
Python3 version 3.6.1 or greater is REQUIRED

Our NNetOneSplit function is located at:
https://github.com/jguzman-tech/sgd-neural-network/blob/master/main.py
** 1. Environment setup
These steps assume a linux system.
*** 1.1. First clone the repo 
git clone https://github.com/jguzman-tech/sgd-neural-network.git
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
prompt$ python3 main.py -h
usage: main.py [-h] [--use-custom-ll] max_epochs step_size n_hidden_units seed

Use the SGD algorithm on the spam.data set

positional arguments:
  max_epochs       The maximum number of epochs
  step_size        The scaling factor used for adjusting weights
  n_hidden_units   The number of hidden parameters in our hidden layer
  seed             The seed used for our random number generator

optional arguments:
  -h, --help       show this help message and exit
  --use-custom-ll  Set this flag if you want to calculate using the LL
                   function we coded.We used the library version to prevent
                   overflow.
#+END_SRC
** 3. Reproduce our results
#+BEGIN_SRC
python3 main.py 500 0.05 10 4
#+END_SRC
The resultant figure will be:
epochs_500_step_0.05_units_10_seed_4_logistic_loss.png

As you can see the filename includes all of the arguments so you can easily
identify them.
