import os
import glob

from utils.prepare_profiling import findReplace # Replace string in every file in a folder


# Get all files inside of `algorithms` folder, to pass them to computation_cost.py through CLI
# algorithms/<FILENAME>.py -> <FILENAME>
ALGORITHMS = [file.split("\\")[1].split('.')[0] for file in glob.glob("algorithms/*.py")]

# Get all files inside of `data` folder, to pass them to computation_cost.py through CLI
# data/<FILENAME>.csv -> <FILENAME>
CSV_FILES = [file.split("\\")[1].split('.')[0] for file in glob.glob("data/*.csv")]

def mprof_commands():
    """
    Generate mprof commands to profile every algorithm on every available dataset.
    Using real world datasets from `CSV_FILES` and generated datasets.

    Return:
        dict of "run" and "plot"
            "run": list mprof run commands
            "plot" list of mprof plot commands

    """

    # Base mprof command, to tune for every algorithm/dataset combination to profile
    base_command = "mprof run --include-children --python python computation_cost.py"

    # Get every available real world dataset
    datasets_commands = CSV_FILES.copy()

    # Filenames to save plots
    filenames = CSV_FILES.copy()

    # Default configuration. For each run, 1 parameter changes.
    default_config = [3, 0.1, 5, 5, 3, 500]

    
    # List on values taken by each parameter
    grid = [
        [2,3,5,7,10,15], # Number of Clusters
        [0.01,0.05,0.1,0.15,0.2,0.3], # Clusters Std
        [1,3,5,10,15,25,50,100], # Number of Numerical Features
        [1,3,5,10,15,25,50,100], # Number of Categorical Features
        [2,3,5,10,15,25,50], # Number of Unique Values taken by categorical features
        [50,100,250,500,1000,1750,2500,5000] # Number of individuals
    ]

    gen_features = ["Number_of_clusters", "Clusters_Std", "Numerical_Features",
                    "Categorical_Features", "Categorical_Uniques", "Number_of_individuals"]

    # List of filenames to save the plots
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            filename = f"{gen_features[i]}_{grid[i][j]}"
            filenames.append(filename)

    # Get every configuration obtained with 1 modification to `default_config`
    gen_configs = []
    for i,g in enumerate(grid):
        for elem in g:
            new_config = default_config.copy()
            new_config[i] = elem
            gen_configs.append(new_config)

    # Prepare for CLI 
    for conf in gen_configs:
        datasets_commands.append(f"generated {' '.join([str(c) for c in conf])}")

    # Full CLI commands 
    run_commands = [
        [f"{base_command} {algorithm} {dataset}" for algorithm in ALGORITHMS] for dataset in datasets_commands
    ]


    # REPLACE BY LIST OF PATH, TO GENERATE A PRETTIER PLOT AND JSON WITH INDICES
    # Commands to plot and save results
    plot_commands = [
        [
            # -t {str(algorithm) + ' ' + str(file)
            f"mprof plot -o computation_cost/{algorithm}/{'_'.join(file.split('_')[:-1])}/{file}.png" for algorithm in ALGORITHMS
        ] for file in filenames
    ]

    # Return a dict
    return {
        "run" : run_commands,
        "plot" : plot_commands
    }


if __name__ == "__main__":
    """
    Profile every Algorithm for every available dataset. Use every dataset in `CSV_FILES`, and profile over
    generated dataset making every parameter vary too.
    Run this file (from repo's root directory) to generate plots in a dedicated folder.

    """
    findReplace("algorithms","#@profile","@profile","*.py")

    gen_features = ["Number_of_clusters", "Clusters_Std", "Numerical_Features",
                    "Categorical_Features", "Categorical_Uniques", "Number_of_individuals"]

    # Create directories for each algorithm's results
    for algorithm in ALGORITHMS:
        for feat in gen_features:
            path = f'computation_cost/{algorithm}/{feat}'
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

    # Get every command to run
    commands = mprof_commands()

    i=0



    ############## RESTART 268

    tot_num = len(commands["run"]) * len(commands["run"][0])
    print(f"Total Number of Commands : {tot_num}")
    # For each algo
    for run, plot in list(zip(commands["run"], commands["plot"])):
        # For each dataset
        for r, p in list(zip(run, plot)):
            i+=1
            #if i < 268:
            #    continue
            print(f"-------------------------------{i}/{tot_num}-------------------------------")
            # Run, Plot, Clean
            # TODO : CHANGE BY check_output(command, timeout=XXX) to set a max run time
            print(f" ... {r} ...")
            os.system(r)
            print(f" ... {p} ...")
            os.system(p)
            os.system("mprof clean")

    findReplace("algorithms","@profile","#@profile","*.py")