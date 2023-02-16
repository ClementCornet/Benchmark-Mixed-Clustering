import os, fnmatch


def findReplace(directory, find, replace, filePattern):
    """
    Replace a sequence of characters in every file in a folder (and subfolders)

    Parameters:
        directory (str) : directory where a the sequence of chars has to be replaced
        find (str) : sequence to replace
        replace (str) : sequence to replace `find` by
        filePattern (str) : regex pattern that the filename has to respect. Example : "*.c" for every c file.

    Example : findReplace("algorithms", "@profile", "#@profile", "*.py")
    In the "algorithms" folder and its subdirectories, replace "@profile" by "#@profile" in every "*.py" file.
    Or simply : comment "@profile" in every python script in the directory.
    
    """


    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)