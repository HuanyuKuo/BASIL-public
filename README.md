# BASIL: BAyesian Selection Inference for Lineage tracking experiment


To analyze DNA barcode read count data from the lineage tracking experiment, this algorithm applies Bayesian filtering method to identify beneficail lineages and infer their selection coefficients.

The key concept is to estimate the **Bayesian probability distribution of size and selection-coefficient** for each lineage. 
At each time step, we infer the population-mean fitness and estimate the indvidual lineage's Bayesian probability.
At the end time, all barcoded lineages are classified to neutral or adaptive group based on their Bayesian estimates.

* Before you run the code, check **System Request** and **Data Setting**. 
* Execute the program 

  ```sh
  python main.py 
  ```
* You can find an example of barcode-data in "./input/Data_BarcodeCount_simuMEE_20220213.txt", with running results under "./output/"
* A simulation program is provided as a data generator. You could play it to test the BASIL analysis. 
* After running, the algorithm will generate (1) Mean fitness trajectory (2) Identified beneficial lineages with their inferred selection coefficients (3) The parametric Bayesian probabilities of all lineages for all time points. 

Here is an example of output image.
<p float="left">
  <img src="/img_README/BASIL_Barcode_Trajectory_Simulation_20220213_v6_ConfidenceFactorBeta=5.00.png" width="700" /> 
</p> 

## System Request
1. python >= 3.6, pystan <= 2.19.1.1
2. Go to the main_scripts folder. Run test_library.py to make sure all required libraries are installed.  
   ```sh
   cd ./BASIL/main_scripts/
   python test_library.py
   ```
    example of output: (You can have different library version)
    ```sh
      Python version 3.9.18 (main, Sep 11 2023, 13:41:44) 
      [GCC 11.2.0]
      Library version
      numpy 1.26.2
      scipy 1.11.4
      matplotlib 3.4.3
      json 2.0.9
      pickle 4.0
      noisyopt 0.2.2
      pystan 2.19.1.1
    ```


3. Generate the Bayesian model by executing stan code under "/main_scripts/model_code/". A new file "SModel_S.pkl" should be generated under the same folder.
    ```sh
    python ./model_code/pystan_SModel_S.py  
    ```
4. This program can use multi processors to speed up MCMC calculations. A multi CPU-core environment is highly recommended. 

## Data Setting & Run BASIL program
1. Run BASIL with provided simulation data. Go to "./main_scripts/". Run BASIL analysis.
    ```sh
    python main.py
    ```
   At the end, it will generate several output files in "./output/".
2. Adjust the following values in "./main_scripts/myConstant.py"  for your case.
    ```sh
    data = FILE NAME OF YOUR BARCODE READ COUNT DATA
    case_name = NAME OF THIS BASIL RUN
    OutputFileDir = DIRECTORY OF OUTPUTFILE
    NUMBER_OF_PROCESSES = ENTER YOUR NUMBER OF PROCESSES
    D = DILUTION FACTOR
    N = CARRYING CAPACITY i.e., total population size before dilution
    ```
3. Test your input data. 
    ```sh
    python myReadfile.py
    ```
4. Run BASIL analysis. 
    ```sh
    python main.py
    ```

## Guide to Package Installation
I run BASIL program with Pycharm under WSL (Windows Subsystem for Linux). This installation guide should work for Windows/Linux user. 

1. First ensure that Pycharm & Anaconda/Miniconda are installed.
2. Open Pycharm and go to the terminal. You will see "(base)" at the head of command line. It's the default environment when you install conda (or anaconda). It is a best practice to avoid installing python packages into the base software environment.
3. To install virtual environment:
    ```sh
     conda create -n myenv-basil python=3.9
    ```
   Replace "myenv-basil" with your desire name.
4. Go to the virtual environment.
    ```sh
     conda activate myenv-basil
    ```
   Now "(myenv-basil)" shows at the head of command line.
5. Download the BASIL program by copying the whole folder or by git clone.
6. Go to the main_scripts folder. Run test_library.py to make sure all required libraries are installed.
   ```sh
   cd ./BASIL-public/main_scripts/
   python test_library.py
   ```
7. Then you'll get error messages of missing package. Install the required package with conda or pip. Rerun the "test_library.py" (step 6) and install all libraries.
   ```sh
   conda install "package-name"
   ```
   Here is mine. 
   ```sh
   conda install numpy
   conda install scipy
   pip install noisyopt
   conda install matplotlib
   conda install pystan=2.19.1.1
   ```


## Debug
1. Most libraries are common in python except of "noisyopt" and "pystan". Both of them are on PyPI. I use Anacaonda (conda) to install libraries on Windows OS. For Linux user, you might use the follow command to install noisyopt and pystan.  
   ```sh
   pip install noisyopt
   python -m pip install pystan
2. If you get error message about pickle protocol (model_load = pickle.load(pickle_file) ValueError: unsupported pickle protocol: 5), it's because your current pickle version is different to the pickle version used for packing the model code. To solve this bug, re-run all model codes to generate new pkl files. 
3. [2022.04.01] Error Message "UnsupportedOperation: fileno" in function sys.stdout.fileno() shows in model colde "pystan_XModel.py" on Windows 10 (IDE spyder4.2.5). This error is solved if I run under external system terminal (Anaoncda Powershell Prompt).

### Folder Description
  **./main_scripts/**: python program of Barcode Filtering Method, 
  **./main_scripts/model_code/**: statistical model underline the Bayesian inferrence
  **./input/**: folder for barcode count data from BLT experiment
  **./ouput/**: folder for outputfile from this python program
  **./simulation_MEE/**: simulation program to generate barcode-count-data. 

### Contact
Huan-Yu Kuo: [linkedin,](https://www.linkedin.com/in/huan-yu-kuo/)  hukuo@ucsd.edu | kuohuanyu@gmail.com

Project Link: [https://github.com/HuanyuKuo/BASIL](https://github.com/HuanyuKuo/Bayesian_Filtering_Method)
