
# Change-Point-Detection
Python3 version 3.6.1 or greater is REQUIRED

** 1. Environment setup
These steps assume a linux system.
*** 1.1. First clone repo
git clone git@github.com/superMicrowave/Change-Point-Detection
*** 1.2. In new terminal, load modules
#+BEGIN_SRC
module load glibc/2.14
#+END_SRC
#+BEGIN_SRC
module load anaconda3
#+END_SRC
**** 1.2.1 avoid lib64 missing
module glibc used for avoiding lib64 missing. you can use command 
#+BEGIN_SRC
ls -l /lib64/libc.so.6
#+END_SRC

to check lib64. If lib64 works, you do not need to load glibc module
*** 1.3. Create environment
#+BEGIN_SRC
conda create -n emacs1 -c conda-forge emacs
#+END_SRC
*** 1.4. Activate environment
#+BEGIN_SRC
conda activate emacs1
#+END_SRC
*** 1.5 Install packages needed for project
#+BEGIN_SRC
conda install pytorch torchvision cpuonly -c pytorch
#+END_SRC
#+BEGIN_SRC
conda install -c anaconda numpy
#+END_SRC
#+BEGIN_SRC
conda install -c conda-forge pandana
#+END_SRC
#+BEGIN_SRC
conda install -c conda-forge matplotlib
#+END_SRC
*** 1.6 Open editor
#+BEGIN_SRC
emacs &
#+END_SRC

** 2. Environment already built
use following command to start environment
#+BEGIN_SRC
module load glibc/2.14
#+END_SRC
#+BEGIN_SRC
module load anaconda3
#+END_SRC
#+BEGIN_SRC
conda activate emacs1
#+END_SRC
#+BEGIN_SRC
emacs &
#+END_SRC

** 3. Execute
*** 3.1 opend accuracy file
*** 3.2 run python
#+BEGIN_SRC
M-x run-python
#+END_SRC
*** 3.3 eval buffer