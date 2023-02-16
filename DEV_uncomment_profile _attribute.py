from utils.prepare_profiling import findReplace

if __name__ == '__main__':

    """
    Uncomment every "@profile" attribute in the repo, to get a more precise time/memory profiling
    """

    findReplace("algorithms","#@profile","@profile","*.py")