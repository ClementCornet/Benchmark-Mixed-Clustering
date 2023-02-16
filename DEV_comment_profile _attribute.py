from utils.prepare_profiling import findReplace

if __name__ == '__main__':

    """
    Comment Every "@profile" attribute in the repo to avoid crashes
    
    """

    findReplace("algorithms","@profile","#@profile","*.py")