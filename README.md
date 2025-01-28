# BASIL: BAyesian Selection Inference for Lineage tracking experiment


This python program aims to analyze DNA barcode read count data from the lineage tracking experiment. 
The algorithm applies Bayesian filtering method to identify beneficail lineages and infer their selection coefficients.

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
* After running, the BASIL program will generate (1) Mean fitness trajectory (2) Identified beneficial lineages with their inferred selection coefficients (3) The parametric Bayesian probabilities of all lineages for all time points
<p float="left">
  <img src="/img_README/BASIL_Barcode_Trajectory_Simulation_20220213_v6_ConfidenceFactorBeta=5.00.png" width="400" />
  <img src="/img_README/meanfitness_trajectory_Bayes_Simulation_20220213_v6.png" width="300" />
</p>

## System Request
1. Python version >= 3.6
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


3. Run all pystan program under "/main_scripts/model_code/". Make sure that "SModel_S.pkl" and "NModel.pkl" files are generated.
    ```sh
    python ./model_code/pystan_SModel_S.py  
    python ./modle_code/pystan_NModel.py
    ```
4. This program can use multi processors to speed up MCMC calculations. A multi CPU-core environment is highly recommended. 

## Data Setting
1. Set parameters. Open "./main_scripts/myConstant.py". Edit **NUMBER_OF_PROCESSES** and **EXPERIMENTAL PARAMETERS** for your case.
    ```sh
    NUMBER_OF_PROCESSES = ENTER YOUR NUMBER OF PROCESSES
    D = ENTER YOUR DILUTION FACTOR
    N = ENTER YOUR CARRYING CAPACITY i.e., total population size before dilution
    ```
2. For input data, place your barcode-count data (txt file) under "./input/", with barcodes=row and time-point=column. The unit of time must be in cycle, not generations.
3. Test your input data. Open "./main_scripts/myReadfile.py". Make **datafilename** as your input file name. 
    ```sh
    datafilename =  YOUR INPUT DATA NAME (barcode-count)
    ```
    Then run "myReadfile.py" to test the file reading.
    ```sh
    python myReadfile.py
    ```
## Run BASIL program
1. Open "./main_scripts/main.py". Edit **datafilename** & **case_name** for your analysis.
    ```sh
    datafilename =  YOUR INPUT DATA NAME (barcode-count)
    case_name = NAME THIS RUNNING CASE
    ```
2. Run BASIL analysis in a command window. 
    ```sh
    python main.py
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
